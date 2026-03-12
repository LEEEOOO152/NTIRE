## 〇、配置环境
```bash
conda env create -f ./config/environment.yml 
conda activate lmf
```
注：若GitHub连接超时，可以注释`config/environment.yml`中的`- git+https://github.com/hiyouga/LlamaFactory.git`以跳过LlamaFactory的安装，或者使用ssh连接并把命令改成`- git+ssh://git@github.com/hiyouga/LlamaFactory.git`

## 一、训练(可选)
0. 配置LlamaFactory
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
```

1. 数据预处理与增强：拷贝训练数据集的`images`文件夹到`data/train`下，运行：
```bash
python ./src/preproc/run_full_pipeline.py \
  --input_json json/original/train_grpo_1536_converted.json \
  --output_dir images/train/processed \
  --local_image_root images/train/images \
  --json_final json/outputs/aug8_swap_crop6_clean.json \
  --num_variations 6 \
  --groups 6
```
获得处理后的图像以及json

2. 数据集配置：填写以下字段到`LlamaFactory\data\dataset_info.json`
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
或请在`file_name`填入`aug8_swap_crop6_clean.json`的绝对路径

3. 启动训练
```bash
cd LlamaFactory
ln -s ../images images
lmf train ../config/training_args.yaml
```


## 二、推理(Validation阶段)
0. 下载训练后权重: 通过网盘分享的文件：checkpoint-960.zip
链接: https://pan.baidu.com/s/1Tg60WPP5atlsWwvg-uz4nA?pwd=3wmp 提取码: 3wmp

1. 类似地，先对推理图像进行预处理
拷贝validation阶段的`image`文件夹到`data/validation`下面，运行
```bash
python ./src/preproc/crop_and_json_pipeline.py \
  --input json/original/inf_validation_phase2.json \
  --output json/outputs/inf_validation_phase2_processed.json \
  --outdir images/validation/processed \
  --groups 1
```
生成处理过后的图像和json

2. 下载`checkpoint-960`到`model`文件夹下，运行以下代码，生成仅answer
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
  --min-checkpoint-step 960 \
  --checkpoint-step-interval 30 \
  --submission-suffix "validation"
```

3. 融合thinking与answer

```bash
python ./src/inf/fuse_and_fix.py \
  --ckpt_paths "src/inf/validation_template.jsonl" \
               "checkpoint-960_validation.jsonl" \
  --ckpt_weights "0,1" \
  --out_fused "json/outputs/fused_intermediate.jsonl" \
  --out_fixed "json/outputs/fused_and_fixed_validation.jsonl" \
  --remove_crops \
  --order_insensitive \
  --strict_images
```
保存最终结果为`checkpoint-960_validation.jsonl`

## 三、推理(Test阶段)
0. 下载训练后权重: 通过网盘分享的文件：checkpoint-960.zip
链接: https://pan.baidu.com/s/1Tg60WPP5atlsWwvg-uz4nA?pwd=3wmp 提取码: 3wmp

1. 类似地，先对推理图像进行预处理
拷贝test阶段的`image`文件夹到`data/test`下面，运行
```bash
python ./src/preproc/crop_and_json_pipeline.py \
  --input json/original/inf_final.json \
  --output json/outputs/inf_final_processed.json \
  --outdir images/test/processed \
  --groups 1
```
生成处理过后的图像和json

2. 生成仅answer
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
  --min-checkpoint-step 960 \
  --checkpoint-step-interval 30 \
  --submission-suffix "test"
```

3. 融合thinking与answer

```bash
python ./src/inf/fuse_and_fix.py \
  --ckpt_paths "src/inf/test_template.jsonl" \
               "checkpoint-960_test.jsonl" \
  --ckpt_weights "0,1" \
  --out_fused "json/outputs/fused_intermediate.jsonl" \
  --out_fixed "json/outputs/fused_and_fixed_test.jsonl" \
  --remove_crops \
  --order_insensitive \
  --strict_images
```
保存最终结果为`checkpoint-960_test.jsonl`

---

**注1：由于数据预处理时采用了随机裁切，所以推理正确率会有一定波动**

**注2：本方法采用thinking模型+answer模型，方便起见thinking模型的输出已经保存为`validation_template.jsonl`和`test_template.jsonl`，以供与answer模型输出结果融合**