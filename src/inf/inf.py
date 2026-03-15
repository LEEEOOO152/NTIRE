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

# ================= 配置区 =================
base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
checkpoint_dir = "models"
# 修改为实际的测试集路径 
test_json_path = "json/outputs/inf_validation_phase2_processed.json"
BATCH_SIZE = 4   
NUM_GPUS = 2      
MAX_pixels = 3538944  
MIN_pixels = 784

# 是否加载 LoRA。若为 False，则直接使用基座模型，不加载任何 checkpoint。
USE_LORA = True

min_checkpoint_step = 960  # 从这个 step 开始测试 (包含)
checkpoint_step_interval = 30  # 每隔多少 step 取一个 checkpoint

PRESETS = {
    "validation": {
        "test_json_path": "json/outputs/inf_validation_phase2_processed.json",
        "submission_suffix": "validation",
    },
    "test": {
        "test_json_path": "json/outputs/inf_final_processed.json",
        "submission_suffix": "test",
    },
}
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for Qwen3-VL with optional LoRA checkpoints")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None, help="Quick preset for json path & suffix")
    parser.add_argument("--base-model-name", default=base_model_name, help="Base model path or hub name")
    parser.add_argument("--checkpoint-dir", default=checkpoint_dir, help="Directory containing LoRA checkpoints")
    parser.add_argument("--test-json-path", default=test_json_path, help="Test JSON path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS, help="Number of GPUs to use")
    parser.add_argument("--max-pixels", type=int, default=MAX_pixels, help="Maximum pixels per image")
    parser.add_argument("--min-pixels", type=int, default=MIN_pixels, help="Minimum pixels per image")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=USE_LORA, help="Whether to load LoRA checkpoints")
    parser.add_argument("--min-checkpoint-step", type=int, default=min_checkpoint_step, help="Minimum checkpoint step to evaluate")
    parser.add_argument("--checkpoint-step-interval", type=int, default=checkpoint_step_interval, help="Checkpoint step interval to evaluate")
    parser.add_argument("--submission-suffix", default="validation", help="Suffix for submission filename")
    return parser.parse_args()

def set_qwen_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_inference(gpu_id, data_chunk, shared_submission_results, shared_failed_cases, lora_ckpt_path, config):
    print(f"进程 {gpu_id} 启动，负责 {len(data_chunk)} 个样本，设备: cuda:{gpu_id}")
    
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    set_qwen_seed(42)

    base_model = config["base_model_name"]
    batch_size = config["batch_size"]
    max_pixels = config["max_pixels"]
    min_pixels = config["min_pixels"]
    use_lora = config["use_lora"]

    # 1. 加载模型与 Processor
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16, 
            device_map={"": device}
        )
        processor = AutoProcessor.from_pretrained(base_model)
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
                turn_from = str(turn.get("from", "")).lower()
                role = "user" if turn_from in {"human", "user"} else "assistant"
                text = turn.get("value", "")
                
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
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=1e-6,
                    top_p=1.0,
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

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_args()

    # Apply preset shortcuts (validation/test)
    if args.preset in PRESETS:
        preset_cfg = PRESETS[args.preset]
        args.test_json_path = preset_cfg["test_json_path"]
        args.submission_suffix = preset_cfg["submission_suffix"]

    base_model_name = args.base_model_name
    checkpoint_dir = args.checkpoint_dir
    test_json_path = args.test_json_path
    BATCH_SIZE = args.batch_size
    NUM_GPUS = args.num_gpus
    MAX_pixels = args.max_pixels
    MIN_pixels = args.min_pixels
    USE_LORA = args.use_lora
    min_checkpoint_step = args.min_checkpoint_step
    checkpoint_step_interval = args.checkpoint_step_interval
    submission_suffix = args.submission_suffix

    runtime_config = {
        "base_model_name": base_model_name,
        "batch_size": BATCH_SIZE,
        "max_pixels": MAX_pixels,
        "min_pixels": MIN_pixels,
        "use_lora": USE_LORA,
    }

    if not os.path.exists(test_json_path):
        print(f"Test file not found: {test_json_path}")
        # 如果是本地测试没有文件，可以注释掉退出
        # exit(1) 
    
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
            # Find checkpoints
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
                
            # Sort by step number assuming format checkpoint-XXXX
            try:
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            except:
                checkpoints.sort()
            
            print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        else:
            # 仅使用基座模型
            checkpoints = [None]
            print("USE_LORA=False: skipping checkpoint loading and using base model only.")

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
                chunk = chunks[rank]
                chunk_list = chunk.tolist() if hasattr(chunk, "tolist") else chunk
                p = Process(
                    target=worker_inference,
                    args=(rank, chunk_list, shared_submission_results, shared_failed_cases, lora_ckpt_path, runtime_config)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            print("\n" + "="*50)
            
            if shared_failed_cases:
                print(f"WARNING: {len(shared_failed_cases)} samples failed for {ckpt_label}.")
                with open(f"failed_inferences_{ckpt_label}.json", 'w', encoding='utf-8') as f:
                    json.dump(list(shared_failed_cases), f, ensure_ascii=False, indent=2)

            # Save submission JSONL
            submission_file = f"{ckpt_label}_{submission_suffix}.jsonl"     #######
            submission_data = list(shared_submission_results)
            
            try:
                submission_data.sort(key=lambda x: str(x['id']))
            except Exception:
                pass

            print(f"Generating {submission_file} with {len(submission_data)} entries...")
            with open(submission_file, 'w', encoding='utf-8') as f:
                for entry in submission_data:
                    # Format per requirements:
                    # {"images": ["./images/xxx", "./images/xxx"], "solution": "..."}
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
