"""Agent 类：把 ReAct 循环封装成一个"有记忆的机器人"

设计要点（写进简历/面试可以讲的四个点）：
1. 上下文与展示分离：messages 是喂给模型的原始上下文，turns 是给人看的历史
2. 一轮对话 = 一次 chat()，工具调用轨迹按轮收集，界面可以按轮折叠展示
3. 步数安全阀 + 工具异常兜底：任何一个工具炸了都不会把整个对话带崩
4. 上下文有界：超出 max_turns 的老对话被压成摘要，token 不会随轮次线性膨胀
"""

import json
import os
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS, TOOL_FUNCS

load_dotenv()

SUMMARY_PROMPT = (
    "把下面的对话压缩成不超过 120 字的中文摘要。"
    "保留：用户的身份与偏好、已经确认的事实、未完成的事项。"
    "丢弃：寒暄、重复内容、工具调用的中间过程。直接输出摘要正文，不要任何前缀或客套。"
)


class Agent:
    """ReAct Agent：能调工具、能多轮对话、能自报"刚才用了什么工具"。"""

    def __init__(
        self,
        system_prompt: str | None = None,
        max_steps: int = 10,
        max_turns: int | None = 6,
        summarize_dropped: bool = True,
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
        self.max_turns = max_turns           # 保留最近几轮原文，None 表示不限制
        self.summarize_dropped = summarize_dropped
        self.system_prompt = system_prompt
        self.on_event = on_event             # 外部观察者：命令打印、界面进度条都靠它

        self.messages: list[dict] = []       # 喂给模型的对话（不含 system，由 _context 拼）
        self.summary: str = ""               # 被裁掉的历史压缩成的文本
        self.turns: list[dict] = []          # 给人看的历史（只含 user / assistant + trace）

    # ---------- 对外接口 ----------
    def chat(self, user_input: str) -> str:
        """用户说一句话 → Agent 思考（可多步调工具）→ 返回最终回答。"""
        self.messages.append({"role": "user", "content": user_input})
        self._maybe_trim()  # 先把上下文压回预算内，再发请求
        trace: list[dict] = []

        for step in range(1, self.max_steps + 1):
            self._emit({"type": "step", "step": step})
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._context(),
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
        self.summary = ""
        self.turns = []

    @property
    def turn_count(self) -> int:
        """已经聊了几轮（展示用）。turns 里一轮 = 一问 + 一答两条。"""
        return sum(1 for t in self.turns if t["role"] == "assistant")

    @property
    def context_size(self) -> int:
        """下一次请求实际会发出去的消息条数（含 system 与摘要）。"""
        return len(self._context())

    # ---------- 上下文管理 ----------
    def _context(self) -> list[dict]:
        """拼装真正发给模型的上下文：system → 摘要 → 最近若干轮原文。

        摘要放在 system 之后、原文之前，模型会把它当作背景知识而不是新指令。
        """
        ctx: list[dict] = []
        if self.system_prompt:
            ctx.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            ctx.append(
                {"role": "system", "content": f"【更早对话的摘要】\n{self.summary}"}
            )
        ctx.extend(self.messages)
        return ctx

    def _maybe_trim(self) -> None:
        """轮次超预算时，从最老的整轮开始丢弃，再把丢掉的部分压成摘要。

        必须按「整轮」切：user 之后可能跟着 assistant(tool_calls) 和若干 tool 消息，
        丢掉其中任何一条都会让 tool_call_id 对不上，接口直接 400。
        """
        if not self.max_turns:
            return
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(starts) <= self.max_turns:
            return

        cut = starts[len(starts) - self.max_turns]
        dropped, self.messages = self.messages[:cut], self.messages[cut:]

        if self.summarize_dropped and dropped:
            self.summary = self._summarize(dropped)
        self._emit(
            {
                "type": "context_trim",
                "dropped": len(dropped),
                "kept_turns": self.max_turns,
                "summary": self.summary,
            }
        )

    def _summarize(self, dropped: list[dict]) -> str:
        """把旧对话压成一段摘要。失败就退化成朴素拼接，绝不因压缩而中断对话。"""
        lines = [
            f"{m['role']}: {m['content']}"
            for m in dropped
            if m.get("content")
        ]
        transcript = "\n".join(lines)
        if self.summary:
            transcript = f"已有摘要：{self.summary}\n\n新增对话：\n{transcript}"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                max_tokens=200,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or transcript[-300:]
        except Exception:
            return transcript[-300:]

    # ---------- 内部 ----------
    def _emit(self, event: dict) -> None:
        """把内部状态变化抛给外部，Agent 自己不 print、不依赖任何界面。"""
        if self.on_event:
            self.on_event(event)

    def _record(self, user_input: str, answer: str, trace: list[dict]) -> None:
        self.turns.append({"role": "user", "content": user_input})
        self.turns.append({"role": "assistant", "content": answer, "trace": trace})
