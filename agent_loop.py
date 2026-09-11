"""阶段2：Agent 循环（ReAct 心脏）

运行：python agent_loop.py
观察：模型会自动分两步——先查时间，再拿结果去计算，
     我们没有写任何"先做什么后做什么"的代码，顺序是模型自己决定的。
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from knowledge import retrieve

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


# ---------- 工具定义 ----------
def calculate(expression: str) -> str:
    """计算一个数学表达式，比如 '2 ** 10'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"


def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_docs(query: str) -> str:
    """检索知识库，返回与问题最相关的几段内容"""
    hits = retrieve(query, top_k=2)
    return "\n---\n".join(doc for doc, score in hits)


# ---------- 工具注册表：JSON 说明 + 函数本体的映射 ----------
TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "检索知识库，返回与问题最相关的几段内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的问题，例如 '什么是 ReAct？'",
                    }
                },
                "required": ["query"],
            },
        },
    },


]

TOOL_FUNCS = {"calculate": calculate, "get_current_time": get_current_time, "search_docs": search_docs}


# ---------- Agent 循环 ----------
if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": "这个项目支持多少种编程语言？",
        }
    ]

    MAX_STEPS = 10  # 安全阀：防止模型无限要工具，烧光余额

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        messages.append(msg)  # 对话历史不能断

        # 模型不再要工具了 → 任务完成，说人话收尾
        if not msg.tool_calls:
            print("最终答案:", msg.content)
            break

        # 模型要工具 → 逐个执行、逐个回传
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"[第{step}步] 模型请求调用: {name}({args})")

            func = TOOL_FUNCS[name]
            result = func(**args)
            print(f"[第{step}步] 本地执行结果: {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
    else:
        print("达到最大步数，强制停止（防失控）")
