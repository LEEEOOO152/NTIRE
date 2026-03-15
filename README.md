## Environment setup
```bash
conda env create -f ./config/environment.yml 
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
pip install git+https://github.com/hiyouga/LlamaFactory.git
conda activate llamaf
```

## 1. Training

1. Data preprocessing & augmentation: copy the training dataset `images` folder to `data/train`, then run:
```bash
python ./src/preproc/train_preproc.py
```
augmentation -> swap -> random crops

2. Dataset config: add the following entry to `LlamaFactory\data\dataset_info.json`
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
3. Replace `LlamaFactory\src\llamafactory\train\sft\trainer.py` with `src\train\trainer.py` to change how the loss is computed.


4. Start training
```bash
cd LlamaFactory
ln -s ../images images
lmf train ../config/training_args.yaml
```


## 2. Validation

Download the trained weights: `checkpoint-960.zip`

Link: https://pan.baidu.com/s/1Tg60WPP5atlsWwvg-uz4nA?pwd=3wmp   Password: 3wmp

1. Preprocess inference images

Copy the validation `image` folder into `data/validation`, then run:
```bash
python ./src/preproc/random_crop.py --preset validation
```
This generates the processed images and JSON.

2. Put `checkpoint-960` under the `model` folder, then run to generate the final result:
```bash
python src/inf/inf.py --preset validation
```


## 3. Test
Download the trained weights: `checkpoint-960.zip`

Link: https://pan.baidu.com/s/1Tg60WPP5atlsWwvg-uz4nA?pwd=3wmp   Password: 3wmp

1. Preprocess inference images

Copy the test `image` folder into `data/test`, then run:
```bash
python ./src/preproc/random_crop.py --preset test
```
This generates the processed images and JSON.

2. Put `checkpoint-960` under the `model` folder, then run to generate the final result:
```bash
python src/inf/inf.py --preset test
```


