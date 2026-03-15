## 0. Environment setup
```bash
conda env create -f ./config/environment.yml 
conda activate lmf
```
Note: If GitHub connection times out, you can comment out `- git+https://github.com/hiyouga/LlamaFactory.git` in `config/environment.yml` to skip installing LlamaFactory, or use SSH by changing it to `- git+ssh://git@github.com/hiyouga/LlamaFactory.git`.

## 1. Training (For ANSWER model)
0. Set up LlamaFactory
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
```

1. Data preprocessing & augmentation: copy the training dataset `images` folder into `data/train`, then run:
```bash
python ./src/preproc/run_full_pipeline.py \
  --input_json json/original/train_grpo_1536_converted.json \
  --output_dir images/train/processed \
  --local_image_root images/train/images \
  --json_final json/outputs/aug8_swap_crop6_clean.json \
  --num_variations 6 \
  --groups 6 \
  --strip-thinking
```
This will generate the processed images and the JSON file. 
**Thinking Tags REMOVED**

2. Dataset configuration: add the following entry to `LlamaFactory\data\dataset_info.json`
```
"ForPhase1_100_aug8_crops_6times_clean": {
    "file_name": "../../json/outputs/aug8_swap_crop6_clean.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "image"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  },
```
Alternatively, set `file_name` to the absolute path of `aug8_swap_crop6_clean.json`.

3. Start training
```bash
cd LlamaFactory
ln -s ../images images
lmf train ../config/training_args.yaml
```

---

## 2. Training (For THINKING model)
1. Data preprocessing & augmentation: copy the training dataset `images` folder into `data/train`, then run:
```bash
python ./src/preproc/run_full_pipeline.py \
  --input_json json/original/train_grpo_1536_converted.json \
  --output_dir images/train/processed \
  --local_image_root images/train/images \
  --json_final json/outputs/aug8_swap_crop6_THINKING.json \
  --num_variations 6 \
  --groups 6
```
This will generate the processed images and the JSON file. 
**Thinking Tags SAVED**

2. Dataset configuration: add the following entry to `LlamaFactory\data\dataset_info.json`
```
"ForPhase1_100_aug8_crops_6times_THINKING": {
    "file_name": "../../json/outputs/aug8_swap_crop6_THINKING.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "image"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  },
```

3. Start training
```bash
cd LlamaFactory
ln -s ../images images
lmf train ../config/training_args_THINKING.yaml
```

---

## 3. Inference (Validation phase)
0. Download the trained weights: shared file `checkpoint-870.zip`
Link：https://pan.quark.cn/s/83702e5e1b50

1. Preprocess inference images
Copy the validation-phase `image` folder into `data/validation`, then run:
```bash
python ./src/preproc/crop_and_json_pipeline.py \
  --input json/original/inf_validation_phase2.json \
  --output json/outputs/inf_validation_phase2_processed.json \
  --outdir images/validation/processed \
  --groups 1
```
This will generate the processed images and JSON.

2. Put `checkpoint-870` under the `models` folder, then run the following to generate **answer-only** output:
```bash
python src/inf/inf_dataset_cvpr_submission_batch.py \
  --base-model-name "Qwen/Qwen3-VL-8B-Instruct" \
  --checkpoint-dir "models" \
  --test-json-path "json/outputs/inf_validation_phase2_processed.json" \
  --batch-size 4 \
  --num-gpus 2 \
  --max-pixels 3538944 \
  --min-pixels 784 \
  --use-lora \
  --min-checkpoint-step 870 \
  --checkpoint-step-interval 30 \
  --submission-suffix "validation"
```

3. Fuse thinking & answer

```bash
python ./src/inf/fuse_and_fix.py \
  --ckpt_paths "src/inf/validation_template.jsonl" \
               "checkpoint-870_validation.jsonl" \
  --ckpt_weights "0,1" \
  --out_fused "json/outputs/fused_intermediate.jsonl" \
  --out_fixed "json/outputs/fused_and_fixed_validation.jsonl" \
  --remove_crops \
  --order_insensitive \
  --strict_images
```
The final output is saved as `checkpoint-870_validation.jsonl`.

## 4. Inference (Test phase)
0. Download the trained weights: shared file `checkpoint-870.zip`
Link：https://pan.quark.cn/s/83702e5e1b50

1. Preprocess inference images
Copy the test-phase `image` folder into `data/test`, then run:
```bash
python ./src/preproc/crop_and_json_pipeline.py \
  --input json/original/inf_final.json \
  --output json/outputs/inf_final_processed.json \
  --outdir images/test/processed \
  --groups 1
```
This will generate the processed images and JSON.

2. Put `checkpoint-870` under the `models` folder, then run the following to generate **answer-only** output:
```bash
python src/inf/inf_dataset_cvpr_submission_batch.py \
  --base-model-name "Qwen/Qwen3-VL-8B-Instruct" \
  --checkpoint-dir "models" \
  --test-json-path "json/outputs/inf_final_processed.json" \
  --batch-size 4 \
  --num-gpus 2 \
  --max-pixels 3538944 \
  --min-pixels 784 \
  --use-lora \
  --min-checkpoint-step 870 \
  --checkpoint-step-interval 30 \
  --submission-suffix "test"
```

3. Fuse thinking & answer

```bash
python ./src/inf/fuse_and_fix.py \
  --ckpt_paths "src/inf/test_template.jsonl" \
               "checkpoint-870_test.jsonl" \
  --ckpt_weights "0,1" \
  --out_fused "json/outputs/fused_intermediate.jsonl" \
  --out_fixed "json/outputs/fused_and_fixed_test.jsonl" \
  --remove_crops \
  --order_insensitive \
  --strict_images
```
The final output is saved as `checkpoint-870_test.jsonl`.
