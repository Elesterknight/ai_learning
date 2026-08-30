from openai import OpenAI
import re
judge_client = OpenAI(api_key='oo', base_url='http://0.0.0.0:8080/v1')
prompt = """
# 任務目標
判斷模型預測答案和標準答案是否一致

# 任務要求
- 兩個答案不必嚴格相等，只要邏輯上等價即可
- 只輸出是或否，不要輸出其他多餘內容

# 模型預測答案
{}

# 標準答案
{}"""
def judge(answer, pre):
    
    messages = [{"role": "user", "content": prompt.format(pre, answer)}]
    output = judge_client.chat.completions.create(
            model="qwen3-235b",
            messages=messages,
            temperature=0.0
        )
        
    model_output = output.choices[0].message.content
    return model_output

def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def format_reward(solution_str):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.match(pattern, solution_str, re.DOTALL)
    return 0.5 if match else 0.0
    

def correctness_reward(solution_str, ground_truth):
    answer = extract_answer(solution_str)
    if '否' in judge(ground_truth, answer):
        return 0.0
    return 1.0



def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    
    format_score = format_reward(solution_str)
    correctness_score = correctness_reward(solution_str, ground_truth)
    
    reward = format_score + correctness_score

    return reward, format_score, correctness_score