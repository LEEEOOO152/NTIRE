import os
import json
import re
import random
import argparse
import numpy as np
import torch
from torch.multiprocessing import Process, Manager, set_start_method
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# ================= 默认配置（可被 CLI 覆盖） =================
DEFAULT_BASE_MODEL = "/public/home/mozhu/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CHECKPOINT_DIR = "/public/home/mozhu/LLaMA-Factory/saves/Qwen3-VL-8B-Instruct/lora/train_2026-03-09-08-23-34"
DEFAULT_TEST_JSON = "/public/home/mozhu/IQAdatasets/ForPhase2/inf_validation_phase2_crops.json"
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_GPUS = 2
DEFAULT_MAX_PIXELS = 3538944
DEFAULT_MIN_PIXELS = 784
DEFAULT_USE_LORA = True
DEFAULT_MIN_CKPT_STEP = 990
DEFAULT_CKPT_INTERVAL = 30
DEFAULT_OUT_PREFIX = "aug8_crops6_clean"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_TEMPERATURE = 1e-6
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with optional LoRA checkpoints (multi-GPU).")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base model name or path")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR, help="LoRA checkpoints root directory")
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON, help="Test JSON path")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per process")
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_NUM_GPUS, help="Number of GPUs / processes")
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Max pixels for processor")
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Min pixels for processor")
    parser.add_argument("--use_lora", action="store_true", default=DEFAULT_USE_LORA, help="Enable LoRA loading")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora", help="Disable LoRA loading")
    parser.set_defaults(use_lora=DEFAULT_USE_LORA)
    parser.add_argument("--min_ckpt_step", type=int, default=DEFAULT_MIN_CKPT_STEP, help="Minimum checkpoint step to include")
    parser.add_argument("--ckpt_interval", type=int, default=DEFAULT_CKPT_INTERVAL, help="Checkpoint step interval")
    parser.add_argument("--out_prefix", default=DEFAULT_OUT_PREFIX, help="Output jsonl filename suffix/prefix")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs (jsonl and failed logs)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Top-p for generation")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens")
    return parser.parse_args()
# ==========================================================

def set_qwen_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_inference(gpu_id, data_chunk, shared_submission_results, shared_failed_cases,
                     lora_ckpt_path, base_model_name, use_lora,
                     batch_size, max_pixels, min_pixels, gen_cfg):
    print(f"进程 {gpu_id} 启动，负责 {len(data_chunk)} 个样本，设备: cuda:{gpu_id}")
    
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    set_qwen_seed(42)

    # 1. 加载模型与 Processor
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16, 
            device_map={"": device}
        )
        processor = AutoProcessor.from_pretrained(base_model_name)
        processor.tokenizer.padding_side = 'left' 

        # 2. 可选加载 LoRA
        if use_lora and lora_ckpt_path:
            model = PeftModel.from_pretrained(model, lora_ckpt_path)
            print(f"[GPU {gpu_id}] Loaded LoRA from {lora_ckpt_path}")
        else:
            print(f"[GPU {gpu_id}] Using base model without LoRA.")

        model.eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {e}")
        return

    # 3. 循环推理
    for i in range(0, len(data_chunk), batch_size):
        batch_samples = data_chunk[i : i + batch_size]
        
        batch_prompts = []
        batch_images = []
        
        for sample in batch_samples:
            image_paths = sample.get("image", [])
            
            # 构建 image context
            image_pool = [{"type": "image", "image": img} for img in image_paths]
            msgs = []
            
            conversations = sample.get("conversations", [])
            
            for turn in conversations:
                role = "user" if turn["from"] == "human" else "assistant"
                text = turn["value"]
                
                if role == "user":
                    content = []
                    text_parts = re.split(r"(<image>|<video>)", text)
                    for seg in text_parts:
                        if seg == "<image>":
                            if image_pool:
                                content.append(image_pool.pop(0))
                        elif seg.strip():
                            content.append({"type": "text", "text": seg.strip()})
                    msgs.append({"role": role, "content": content})
                else:
                    msgs.append({"role": role, "content": [{"type": "text", "text": text}]})
            
            # Qwen3-VL 只需要输入对话历史，它会自动生成 next token
            # 确保不包含最后的 Assistant 回答（如果有的话，应该是在做 Training/Validating）
            # 对于纯测试，一般 conversations 最后一条是 User 的提问
            # 这里保守起见，如果最后一条是 assistant，我们去掉它？
            # 这里的逻辑是直接给 input_msgs 给 apply_chat_template(..., add_generation_prompt=True)
            # 如果 conversations 包含了 User 和 Assistant 的多轮对话，且我们希望接着生成：
            # 如果最后一条是 Human，那么 add_generation_prompt=True 会加上 Assistant: 引导
            
            # 使用前两条作为 Prompt (System/User)，或者保持所有 User 提出的内容
            # 原代码是 msgs[:2]，这通常是 System + User (with images)
            input_msgs = msgs[:2] 

            text_input = processor.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)
            
            batch_prompts.append(text_input)
            batch_images.append(image_paths) 

        if not batch_prompts: continue

        try:
            inputs = processor(
                text=batch_prompts, 
                images=batch_images, 
                padding=True, 
                return_tensors="pt", 
                max_pixels=max_pixels,
                min_pixels=min_pixels
            ).to(device)
            
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    do_sample=True,
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            input_len = inputs.input_ids.shape[1]
            output_texts = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)
            
            for j, output_text in enumerate(output_texts):
                sample_id = batch_samples[j]["id"]
                sample_images = batch_samples[j].get("image", [])
                print(f"[GPU {gpu_id}] Inference done for {sample_id}")
                shared_submission_results.append({
                    "id": sample_id, 
                    "response": output_text,
                    "images": sample_images
                })

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing batch: {str(e)}")
            for sample in batch_samples:
                shared_failed_cases.append({"id": sample["id"], "error": str(e)})
            continue

    print(f"进程 {gpu_id} 推理完成。")

def main():
    args = parse_args()

    base_model_name = args.base_model
    checkpoint_dir = args.checkpoint_dir
    test_json_path = args.test_json
    batch_size = args.batch_size
    NUM_GPUS = args.num_gpus
    MAX_pixels = args.max_pixels
    MIN_pixels = args.min_pixels
    USE_LORA = args.use_lora
    min_checkpoint_step = args.min_ckpt_step
    checkpoint_step_interval = args.ckpt_interval
    out_prefix = args.out_prefix
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    gen_cfg = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if not os.path.exists(test_json_path):
        print(f"Test file not found: {test_json_path}")
        return
    
    try:
        with open(test_json_path, 'r', encoding='utf-8') as f:
            all_test_data = json.load(f)
        
        total_samples = len(all_test_data)
        print(f"Total samples: {total_samples}, distributing to {NUM_GPUS} GPUs...")

        if total_samples < NUM_GPUS:
            chunks = [all_test_data]
            active_gpus = 1
        else:
            chunks = np.array_split(all_test_data, NUM_GPUS)
            active_gpus = NUM_GPUS
        
        if USE_LORA:
            all_checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
            checkpoints = []
            for ckpt in all_checkpoints:
                try:
                    step = int(ckpt.split('-')[-1])
                    if step >= min_checkpoint_step and (step - min_checkpoint_step) % checkpoint_step_interval == 0:
                        checkpoints.append(ckpt)
                except ValueError:
                    print(f"Skipping malformed checkpoint name: {ckpt}")
                    continue
            try:
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            except:
                checkpoints.sort()
            print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        else:
            checkpoints = [None]
            print("USE_LORA=False: using base model only.")

        for ckpt_name in checkpoints:
            lora_ckpt_path = os.path.join(checkpoint_dir, ckpt_name) if ckpt_name else None
            ckpt_label = ckpt_name if ckpt_name else "base_model"
            print(f"\n{'='*20} Processing Checkpoint: {ckpt_label} {'='*20}")
            if lora_ckpt_path:
                print(f"Path: {lora_ckpt_path}")

            manager = Manager()
            shared_submission_results = manager.list()
            shared_failed_cases = manager.list()

            processes = []
            for rank in range(active_gpus):
                p = Process(target=worker_inference, args=(
                    rank,
                    chunks[rank].tolist(),
                    shared_submission_results,
                    shared_failed_cases,
                    lora_ckpt_path,
                    base_model_name,
                    USE_LORA,
                    batch_size,
                    MAX_pixels,
                    MIN_pixels,
                    gen_cfg,
                ))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            print("\n" + "="*50)
            
            if shared_failed_cases:
                print(f"WARNING: {len(shared_failed_cases)} samples failed for {ckpt_label}.")
                failed_path = os.path.join(out_dir, f"failed_inferences_{ckpt_label}.json")
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(list(shared_failed_cases), f, ensure_ascii=False, indent=2)

            submission_file = os.path.join(out_dir, f"{ckpt_label}_{out_prefix}.jsonl")
            submission_data = list(shared_submission_results)
            try:
                submission_data.sort(key=lambda x: str(x['id']))
            except Exception:
                pass

            print(f"Generating {submission_file} with {len(submission_data)} entries...")
            with open(submission_file, 'w', encoding='utf-8') as f:
                for entry in submission_data:
                    formatted_images = [f"./images/{os.path.basename(img)}" for img in entry.get('images', [])]
                    solution = entry.get('response', "")
                    formatted_entry = {
                        "images": formatted_images,
                        "solution": solution
                    }
                    f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
            print(f"Done. Saved to {submission_file}")
    
    except Exception as e:
        print(f"Main process error: {e}")


if __name__ == "__main__":
    main()
