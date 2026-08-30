# 使用方法

## 下載模型及資料
### 下載qwen2.5-0.5b和siglip
qwen2.5-0.5b: \
https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct \
siglip: \
此處使用的是如下版本的siglip（模型小，但是效果可能沒那麼好，訓練更快，顯示記憶體要求更低）：\
https://hf-mirror.com/google/siglip-base-patch16-224

也可以使用效果更好的版本，但是模型會更大（注意，使用這個版本可能需要修改image_pad_num這個參數，這個版本的模型輸出的圖片特徵為（b,729,dim），在圖片壓縮的時候是reshape成（b,729/9,dim*9））：\
https://hf-mirror.com/google/siglip-so400m-patch14-384

### 下載資料集
1、預訓練資料：\
圖片資料：\
https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K \
中文文本資料：\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions \
2、SFT資料:\
圖片資料:\
https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset \
中文文本資料:\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions

## 開始訓練
### 直接執行
預訓練:\
python train.py\
SFT:\
python sft_train.py
### torchrun
預訓練:\
torchrun --nproc_per_node=2 train.py
SFT:\
torchrun --nproc_per_node=2 sft_train.py
### deepspeed
預訓練:\
deepspeed --include 'localhost:0,1' train.py\
SFT:\
deepspeed --include 'localhost:0,1' sft_train.py

## 測試
python test.py
