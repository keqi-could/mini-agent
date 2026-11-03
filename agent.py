"""Agent 类：把 ReAct 循环封装成一个"有记忆的机器人"

设计要点（写进简历/面试可以讲的三个点）：
1. 上下文与展示分离：messages 是喂给模型的原始上下文，turns 是给人看的历史
2. 一轮对话 = 一次 chat()，工具调用轨迹按轮收集，界面可以按轮折叠展示
3. 步数安全阀 + 工具异常兜底：任何一个工具炸了都不会把整个对话带崩
"""

import json
import os
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS, TOOL_FUNCS

load_dotenv()


class Agent:
    """ReAct Agent：能调工具、能多轮对话、能自报"刚才用了什么工具"。"""

    def __init__(
        self,
        system_prompt: str | None = None,
        max_steps: int = 10,
        model: str = "deepseek-chat",
        tools: list | None = None,
        tool_funcs: dict | None = None,
        client: OpenAI | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.client = client or OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self.model = model
        self.tools = TOOLS if tools is None else tools
        self.tool_funcs = TOOL_FUNCS if tool_funcs is None else tool_funcs
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.on_event = on_event  # 外部观察者：命令行打印、界面进度条都靠它

        self.messages: list[dict] = []  # 喂给模型的上文（含 system / tool 消息）
        self.turns: list[dict] = []     # 给人看的历史（只含 user / assistant + trace）

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    # ---------- 对外接口 ----------
    def chat(self, user_input: str) -> str:
        """用户说一句话 → Agent 思考（可多步调工具）→ 返回最终回答。"""
        self.messages.append({"role": "user", "content": user_input})
        trace: list[dict] = []

        for step in range(1, self.max_steps + 1):
            self._emit({"type": "step", "step": step})
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
            )
            msg = resp.choices[0].message

            # 模型自己说的话必须原样回填（尤其是 tool_calls），否则下一步它会"失忆"
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:  # 不再要工具 → 这一轮结束
                answer = msg.content or ""
                self._record(user_input, answer, trace)
                self._emit({"type": "final", "answer": answer, "steps": step})
                return answer

            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                self._emit({"type": "tool_start", "step": step, "tool": name, "args": args})

                try:
                    result = str(self.tool_funcs[name](**args))
                except KeyError:
                    result = f"没有名为 {name} 的工具"
                except Exception as e:  # 工具内部报错也不该中断对话
                    result = f"工具执行失败: {e}"

                trace.append({"step": step, "tool": name, "args": args, "result": result})
                self._emit(
                    {"type": "tool_end", "step": step, "tool": name, "result": result}
                )
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        # 走满 max_steps 还没收敛：强制刹车，防止无限调工具烧钱
        fallback = "（达到最大步数，已强制停止。可以把问题拆小一点再问）"
        self._record(user_input, fallback, trace)
        self._emit({"type": "max_steps", "steps": self.max_steps})
        return fallback

    def reset(self) -> None:
        """清空记忆，回到刚创建时的状态。"""
        self.messages = []
        self.turns = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    @property
    def turn_count(self) -> int:
        """已经聊了几轮（展示用）。turns 里一轮 = 一问 + 一答两条。"""
        return sum(1 for t in self.turns if t["role"] == "assistant")

    # ---------- 内部 ----------
    def _emit(self, event: dict) -> None:
        """把内部状态变化抛给外部，Agent 自己不 print、不依赖任何界面。"""
        if self.on_event:
            self.on_event(event)

    def _record(self, user_input: str, answer: str, trace: list[dict]) -> None:
        self.turns.append({"role": "user", "content": user_input})
        self.turns.append({"role": "assistant", "content": answer, "trace": trace})
