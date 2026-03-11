## 〇、配置LlamaFactory
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
conda env create -f ./config/environment.yml #### 缺少cv2等。。
conda activate lmf
```

## 一、数据预处理、配置与训练(可选)
1. 数据预处理与增强：拷贝训练数据集的`images`文件夹到 `data/train` 下，运行：
```bash
python ./src/preproc/run_full_pipeline.py \
  --input_json json/original/train_grpo_1536_converted.json \
  --output_dir images/train/processed \
  --local_image_root images/train/images \
  --json_final json/output/aug8_swap_crop6_clean.json \
  --num_variations 6 \
  --groups 6
```
获得处理后的图像以及json

2. 数据集配置：填写以下字段到`LlamaFactory\data\dataset_info.json`
```
"ForPhase1_100_aug8_crops_6times_clean": {
    "file_name": "aug8_swap_crop6_clean.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "image"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  },
```

3. 启动训练：拷贝`config/training_args.yaml`到`LlamaFactory/examples/train_lora`下面
```bash
cd LlamaFactory
lmf train ../config/training_args.yaml
```
开启训练

## 二、推理
1. 类似地，先对推理图像进行预处理
拷贝validation的`image`文件夹到`data/validation`下面，运行
```bash
python ./src/preproc/crop_and_json_pipeline.py \
  --input json/original/inf_validation_phase2.json \
  --output json/output/inf_validation_phase2_processed.json \
  --outdir data/validation/processed \
  --groups 1
```
生成处理过后的图像和json

2. 生成仅answer
```bash
python "src/inf/inf_dataset_cvpr_submission_batch.py" \
  --base_model Qwen/Qwen3-VL-8B-Instruct \
  --checkpoint_dir models/ \ 
  --test_json json/output/inf_validation_phase2_processed.json \
  --batch_size 4 \
  --num_gpus 2 \
  --use_lora \
  --min_ckpt_step 960 \
  --ckpt_interval 30 \
  --out_prefix myrun \
  --temperature 1e-6 \
  --top_p 1.0 \
  --max_new_tokens 512
  --out_dir json/output
```

3. 融合thinking与answer

```bash
python ./src/inf/fuse_and_fix.py `
  --ckpt_paths "src\inf\validation_template.jsonl" `
               "" `
  --ckpt_weights "0,1" `
  --out_fused "json/output/fused_intermediate.jsonl" `
  --out_fixed "json/output/fused_and_fixed.jsonl" `
  --remove_crops `
  --order_insensitive `
  --strict_images
```