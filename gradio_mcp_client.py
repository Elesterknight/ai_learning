import gradio as gr
from mcp.client.sse import sse_client
from mcp import ClientSession
from openai import AsyncOpenAI
import json


SYSTEM_PROMPT = """你是一個AI助手。
你可以使用 MCP 伺服器提供的工具來完成任務。
MCP 伺服器會動態提供工具，你需要先檢查當前可用的工具。

在使用 MCP 工具時，請遵循以下步驟：
1、根據任務需求選擇合適的工具
2、按照工具的參數要求提供正確的參數
3、觀察工具的回傳結果，並根據結果決定下一步操作
4、工具可能會發生變化，比如新增工具或現有工具消失

請遵循以下指南：
- 使用工具時，確保參數符合工具的文檔要求
- 如果出現錯誤，請理解錯誤原因並嘗試用修正後的參數重新呼叫
- 按照任務需求逐步完成，優先選擇最合適的工具
- 如果需要連續呼叫多個工具，請一次只呼叫一個工具並等待結果

請清楚地向使用者解釋你的推理過程和操作步驟。
"""
     
async def query(query: str, mcp_server_url, model_name, base_url, api_key, temperature):
    
    client = AsyncOpenAI(
            base_url=base_url, api_key=api_key
        )

    async with sse_client(mcp_server_url) as streams:
    
        async with ClientSession(*streams) as session:

            await session.initialize()
            
            response = await session.list_tools()
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]

            
            # 初始化 LLM API 呼叫
            response = await client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
                tools=available_tools,
                stream=True
            )
            # message = response.choices[0].message
            full_response = ""
            tool_call_text = ""
          
            while True:
                func_call_list = []
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield tool_call_text, full_response.replace('<think>', "").replace('</think>', "")  # 流式回傳當前累積內容
                    elif chunk.choices[0].delta.tool_calls:
                        
                        for tcchunk in chunk.choices[0].delta.tool_calls:
                            if len(func_call_list) <= tcchunk.index:
                                func_call_list.append({
                                    "id": "",
                                    "name": "",
                                    "type": "function", 
                                    "function": { "name": "", "arguments": "" } 
                                })
                            tc = func_call_list[tcchunk.index]
                            if tcchunk.id:
                                tc["id"] += tcchunk.id
                            if tcchunk.function.name:
                                tc["function"]["name"] += tcchunk.function.name
                            if tcchunk.function.arguments:
                                tc["function"]["arguments"] += tcchunk.function.arguments
                
                        
                if not func_call_list:
                    break
                
                full_response += '🛠️ 呼叫工具...\n'
                yield tool_call_text, full_response.replace('<think>', "").replace('</think>', "")
                
                for tool_call in func_call_list:
                    print(tool_call)
                    tool_name = tool_call['function']['name']
                    if tool_call['function']['arguments']:
                        tool_args = json.loads(tool_call['function']['arguments'])
                    else:
                        tool_args = {}

                    # 執行工具呼叫
                    result = await session.call_tool(tool_name, tool_args)
                    # 記錄呼叫詳情到狀態列
                    tool_call_text += f"✅ 工具回傳: {tool_name}\n參數: {tool_args}\n結果: {str(result.content)}\n---\n"
                    yield tool_call_text, full_response.replace('<think>', "").replace('</think>', "")  # 先更新狀態列
                    
                    # 將工具呼叫和結果新增到訊息歷史
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call['id'],
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args)
                                }
                            }
                        ]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": str(result.content)
                    })

                # 將工具呼叫的結果交給 LLM
                response = await client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    messages=messages,
                    tools=available_tools,
                    stream=True)
            
                

with gr.Blocks() as demo:
    gr.Markdown("## MCP 客戶端")
    
    # 左右分欄佈局
    with gr.Row():
        # 左側參數輸入欄
        with gr.Column(scale=1):
            gr.Markdown("### 🧠 大模型設定")
            model_name = gr.Textbox(
                label="模型名稱"
            )
            base_url = gr.Textbox(
                label="API 地址"
            )
            api_key = gr.Textbox(
                label="API Key",
                type="password"
            )
            temperature = gr.Number(
                label="溫度",
                value=0.0,
            )
            
            gr.Markdown("### 🌐 MCP 服務設定")
            mcp_server_url = gr.Textbox(
                label="MCP 服務地址"
            )
            
            # 工具呼叫狀態面板
            tool_status = gr.Textbox(
                label="🛠️ 工具呼叫記錄",
                lines=10,
                interactive=False,
                autoscroll=True,
            )

        # 右側輸出區域
        with gr.Column(scale=2):
            gr.Markdown("### 💬 交互窗口")
            result_display = gr.Textbox(
                label="🧠 模型輸出",
                lines=35,
                show_copy_button=True,
            )
    
    # 最底部問題輸入區
    with gr.Row():
        query_input = gr.Textbox(
            label="❓ 輸入你的問題",
            placeholder="輸入問題後點擊生成按鈕...",
            scale=4
        )
        generate = gr.Button(
            "🚀 開始生成",
            scale=1,
            variant="primary"
        )
    
    generate.click(fn=query, inputs=[query_input, mcp_server_url, model_name, base_url, api_key, temperature], outputs=[tool_status, result_display])
    

    
    
    
if __name__ == "__main__":
    demo.queue().launch(server_name='0.0.0.0', allowed_paths=['./'])

