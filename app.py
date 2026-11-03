"""阶段4：给 Agent 套上网页对话界面（阶段5 已接入 Agent 类）

运行（注意：不是 python app.py！）：
    streamlit run app.py

浏览器会自动打开 http://localhost:8501
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_key(name: str) -> str:
    """本地 .env 优先；部署到 Streamlit Cloud 时从 st.secrets 读取。"""
    value = os.environ.get(name)
    if not value and hasattr(st, "secrets"):
        value = st.secrets.get(name)
    if not value:
        raise ValueError(f"缺少 {name}，请在 .env（本地）或 Streamlit secrets（云端）中配置")
    os.environ[name] = value
    return value


# 两个 Key 都必须在导入工具/知识库之前注入环境变量
_get_key("DEEPSEEK_API_KEY")
_get_key("ZHIPU_API_KEY")

from agent import Agent  # noqa: E402  （必须在 Key 注入之后导入）

# ---------- 系统提示词：约束模型"只根据资料回答"（防幻觉的关键一步） ----------
SYSTEM_PROMPT = (
    "你是 mini-agent 的知识库助手。回答必须基于 search_docs 检索到的内容；"
    "如果检索结果不足以回答，直接说'知识库中没有相关信息'，不要编造。"
)


# ================= 网页部分 =================
st.title("mini-agent 知识库助手")
st.caption("从零手写的 Agent：ReAct 循环 + 工具调用 + RAG 检索，不依赖任何 Agent 框架")

# Agent 实例存在 session_state 里：Streamlit 每次交互都重跑脚本，
# 靠这个对象把「对话记忆」跨请求留下来（局部变量会被重跑清空）
if "agent" not in st.session_state:
    st.session_state.agent = Agent(system_prompt=SYSTEM_PROMPT)

agent: Agent = st.session_state.agent

# ---------- 侧边栏：观察 Agent 的记忆状态 ----------
with st.sidebar:
    st.subheader("Agent 状态")
    st.metric("已完成轮次", agent.turn_count)
    st.caption(f"上下文消息数：{len(agent.messages)} 条（含 system 与工具回传）")
    if st.button("清空对话记忆", use_container_width=True):
        agent.reset()
        st.rerun()
    st.divider()
    st.caption("同一会话内 Agent 会记住你之前说过的话，可以试试「我叫小明」→「我叫什么？」")

# ---------- 画出历史对话（数据来自 Agent 的 turns，不是单独维护的列表） ----------
for turn in agent.turns:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        trace = turn.get("trace") or []
        if trace:
            # 把命令行里的 print 搬上网页：每一轮的工具调用过程可折叠查看
            with st.expander(f"🔍 这一轮用了 {len(trace)} 次工具调用"):
                for item in trace:
                    st.code(
                        f"[第{item['step']}步] {item['tool']}({item['args']})\n"
                        f"结果: {item['result']}",
                        language=None,
                    )

# ---------- 输入框（固定在页面底部） ----------
user_input = st.chat_input("问点什么，比如：什么是 ReAct？")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中（查知识库约需 5~10 秒）..."):
            answer = agent.chat(user_input)

        st.markdown(answer)

        trace = agent.turns[-1].get("trace") or []
        if trace:
            with st.expander(f"🔍 这一轮用了 {len(trace)} 次工具调用"):
                for item in trace:
                    st.code(
                        f"[第{item['step']}步] {item['tool']}({item['args']})\n"
                        f"结果: {item['result']}",
                        language=None,
                    )
