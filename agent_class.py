"""阶段5-第1课：把 Agent 循环重构成类

运行：python agent_class.py
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from knowledge import retrieve
from agent_loop import calculate, get_current_time, search_docs, TOOLS, TOOL_FUNCS

load_dotenv()


class Agent:
    """一个能调用工具、多轮对话的 ReAct Agent"""

    def __init__(self, max_steps: int = 10):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self.messages = []        # 对话记忆，跟着这个 Agent 走
        self.max_steps = max_steps

    def chat(self, user_input: str) -> str:
        """用户说一句话，Agent 思考（可能调工具）后回答"""
        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=TOOLS,
            )
            msg = resp.choices[0].message
            self.messages.append(msg)

            if not msg.tool_calls:
                return msg.content     # 想通了，直接把答案还回去

            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                print(f"[第{step}步] 调用: {name}({args})")

                # TODO 1：模仿 agent_loop.py 第 128~129 行，
                # 从 TOOL_FUNCS 里取出函数并执行，结果存到 result 变量
                result = TOOL_FUNCS[name](**args)

                print(f"[第{step}步] 结果: {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

        return "达到最大步数，强制停止（防失控）"


if __name__ == "__main__":
    agent = Agent()                              # 造一个机器人（只造一次！）

    print(agent.chat("我叫小明，今年 20 岁"))      # 第 1 句：自我介绍
    print(agent.chat("我叫什么名字？"))             # 第 2 句：考它记性
    print(agent.chat("我今年多大？"))               # 第 3 句：再考一次

