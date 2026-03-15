## 配置环境
```bash
conda env create -f ./config/environment.yml 
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
pip install git+https://github.com/hiyouga/LlamaFactory.git
conda activate llamaf
```

## 一、训练
1. 数据预处理与增强：拷贝训练数据集的`images`文件夹到`data/train`下，运行：
```bash
python ./src/preproc/train_preproc.py
```

2. 数据集配置：填写以下字段到`LlamaFactory\data\dataset_info.json`
```
"NTIRE_train": {
    "file_name": "../../json/outputs/train_aug.json",
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

3. 启动训练
```bash
cd LlamaFactory
ln -s ../images images
lmf train ../config/training_args.yaml
```


## 二、Validation
0. 下载训练后权重: 通过网盘分享的文件：checkpoint-960.zip
链接: https://pan.baidu.com/s/1Tg60WPP5atlsWwvg-uz4nA?pwd=3wmp 提取码: 3wmp

1. 类似地，先对推理图像进行预处理
拷贝validation阶段的`image`文件夹到`data/validation`下面，运行
```bash
python ./src/preproc/random_crop.py --preset validation
```
生成处理过后的图像和json

2. 下载`checkpoint-960`到`model`文件夹下，运行以下代码，生成answer
```bash
python src/inf/inf.py --preset validation
```

3. 融合thinking与answer

```bash
python ./src/inf/fuse_and_fix.py --preset validation
```
输出文件名为`checkpoint-960_validation.jsonl`

## 三、Test
0. 下载训练后权重: 通过网盘分享的文件：checkpoint-960.zip
链接: https://pan.baidu.com/s/1Tg60WPP5atlsWwvg-uz4nA?pwd=3wmp 提取码: 3wmp

1. 类似地，先对推理图像进行预处理
拷贝test阶段的`image`文件夹到`data/test`下面，运行
```bash
python ./src/preproc/random_crop.py --preset test
```
生成处理过后的图像和json

2. 生成仅answer
```bash
python src/inf/inf.py --preset test
```

3. 融合thinking与answer

```bash
python ./src/inf/fuse_and_fix.py --preset test
```
输出文件名为`checkpoint-960_test.jsonl`

---
