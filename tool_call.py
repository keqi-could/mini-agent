"""阶段1：手写工具调用（Function Calling）

运行：python tool_call.py
观察：模型不会自己算 123*456*7，而是请求调用我们的 calculate 工具。
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# ---------- ① 定义工具：就是普通的 Python 函数 ----------
def calculate(expression: str) -> str:
    """计算一个数学表达式，比如 '2 ** 10'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"


# ---------- ② 用 JSON Schema 把工具"说明书"写给模型看 ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除、乘方等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '2 ** 10'",
                    }
                },
                "required": ["expression"],
            },
        },
    },{
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},   # 这个工具不需要参数，所以是空的
                "required": [],
            },
        },
    }
]


# ---------- ③ 第一轮：问题 + 工具清单一起发出去 ----------
messages = [{"role": "user", "content": "现在几点了?"}]

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)
msg = resp.choices[0].message

print("模型的文字回复:", msg.content)
print("模型的工具调用请求:", msg.tool_calls)


# ---------- ④ 模型想用工具？我们本地执行 ----------
if msg.tool_calls:
    call = msg.tool_calls[0]
    print("要调用的函数名:", call.function.name)
    print("参数（原始 JSON 字符串）:", call.function.arguments)

    args = json.loads(call.function.arguments)  # JSON 字符串 -> 字典
    available_tools = {"calculate": calculate, "get_current_time": get_current_time}
    func = available_tools[call.function.name]
    result = func(**args)

    print("本地执行结果:", result)

    # ---------- ⑤ 把工具结果回传，让模型给出最终答案 ----------
    messages.append(msg)  # 记录模型"想调用工具"这件事（对话历史不能断）
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        }
    )

    resp2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    print("最终答案:", resp2.choices[0].message.content)
