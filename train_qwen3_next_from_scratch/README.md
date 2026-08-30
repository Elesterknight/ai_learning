## 訓練
### 直接執行
預訓練:\
python pretrain.py\
SFT:\
python sft_train.py
### torchrun
預訓練:\
torchrun --nproc_per_node=2 pretrain.py\
SFT:\
torchrun --nproc_per_node=2 sft_train.py
### deepspeed
預訓練:\
deepspeed --include 'localhost:0,1' pretrain.py\
SFT:\
deepspeed --include 'localhost:0,1' sft_train.py

## 測試
test_moe.ipynb