# 使用方法

## 下載資料

https://github.com/jingyaogong/minimind
![image](.\screenshot-20241207-093824.png)

## 開始訓練
### 直接執行
預訓練:\
python moe_train.py\
SFT:\
python moe_sft_train.py
### torchrun
預訓練:\
torchrun --nproc_per_node=2 moe_train.py
SFT:\
torchrun --nproc_per_node=2 moe_sft_train.py
### deepspeed
預訓練:\
deepspeed --include 'localhost:0,1' moe_train.py\
SFT:\
deepspeed --include 'localhost:0,1' moe_sft_train.py

## 測試
python moe_test.py