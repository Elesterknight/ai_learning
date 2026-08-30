import re
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
def correctness_reward(prompts, responses, answers):
    
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"問題:\n{prompts[0]}", f"\n答案:\n{answers[0]}", f"\n模型輸出:\n{responses[0]}", f"\n提取後的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answers)]

# 生成答案是否是數字的獎勵（單純依賴結果是否正確進行獎勵，條件很苛刻，會導致獎勵比較稀疏，模型難以收斂，所以加上答案是否是數字的獎勵，雖然答案錯誤，但是至少生成的是數字（對於數學問題），也要給予適當獎勵）
def digit_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

# 格式獎勵
def hard_format_reward(prompts, responses, answers):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]

# 標記獎勵（改善格式獎勵稀疏問題）
def mark_reward(prompts, responses, answers):
    return [mark_num(response) for response in responses]