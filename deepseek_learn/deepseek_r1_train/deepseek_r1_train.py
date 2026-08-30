import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>
"""
def process_data(data):
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    }) 
    return data
def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

# 生成答案是否正確的獎勵
def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"問題:\n{prompts[0][-1]['content']}", f"\n答案:\n{answer[0]}", f"\n模型輸出:\n{responses[0]}", f"\n提取後的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answer)]
# 生成答案是否是數字的獎勵（單純依賴結果是否正確進行獎勵，條件很苛刻，會導致獎勵比較稀疏，模型難以收斂，所以加上答案是否是數字的獎勵，雖然答案錯誤，但是至少生成的是數字（對於數學問題），也要給予適當獎勵）
def digit_reward(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

# 格式獎勵
def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]
# 格式獎勵
def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]
# 標記獎勵（改善格式獎勵稀疏問題）
def mark_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]


if __name__ == '__main__':
    model_name = "/home/user/Downloads/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    # 如果使用lora方法訓練，取消如下注釋
    # lora_config = LoraConfig(
    # r=8,  
    # lora_alpha=256,  
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # lora_dropout=0.1, 
    # task_type=TaskType.CAUSAL_LM)
    # # 使用lora方法訓練
    # model = get_peft_model(model, lora_config)
    model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ds = load_dataset('/home/user/wyf/deepseek_learn/gsm8k_chinese')
    data = process_data(ds['train'])
    
    output_dir="output"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )
    
    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        mark_reward,
        soft_format_reward,
        hard_format_reward,
        digit_reward,
        correctness_reward
        ],
    args=training_args,
    train_dataset=data,

)
    trainer.train()
    trainer.save_model(output_dir)
