# 使用方法

## 下載資料

https://github.com/jingyaogong/minimind
![image](.\screenshot-20241207-093824.png)

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
test_llm.ipynb