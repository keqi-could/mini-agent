"""阶段4：给 Agent 套上网页对话界面

运行（注意：不是 python app.py！）：
    streamlit run app.py

浏览器会自动打开 http://localhost:8501
"""

import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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


client = OpenAI(
    api_key=_get_key("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 智谱 Key 也提前注入环境变量，knowledge.py import 时会用到
_get_key("ZHIPU_API_KEY")

from knowledge import retrieve


# ---------- 系统提示词：约束模型"只根据资料回答"（防幻觉的关键一步） ----------
SYSTEM_PROMPT = (
    "你是 mini-agent 的知识库助手。回答必须基于 search_docs 检索到的内容；"
    "如果检索结果不足以回答，直接说'知识库中没有相关信息'，不要编造。"
)


# ---------- 工具定义（和 agent_loop.py 相同的三件套） ----------
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
            "parameters": {"type": "object", "properties": {}, "required": []},
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

TOOL_FUNCS = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_docs": search_docs,
}


# ---------- Agent 循环：包成函数，返回 (最终答案, 工具调用轨迹) ----------
def run_agent(messages):
    trace = []  # 记录这次用了哪些工具，好在页面上展示
    for step in range(1, 11):  # MAX_STEPS = 10，安全阀不丢
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # 统一转成 dict 存（方便保存、也方便页面重画）
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls,
            }
        )

        if not msg.tool_calls:  # 模型不再要工具 → 任务完成
            return msg.content, trace

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            result = TOOL_FUNCS[name](**args)
            trace.append(f"{name}({args})\n结果: {result}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
    return "（达到最大步数，强制停止）", trace


# ================= 网页部分 =================
st.title("mini-agent 知识库助手")
st.caption("从零手写的 Agent：ReAct 循环 + 工具调用 + RAG 检索，不依赖任何 Agent 框架")

# 历史消息存在 session_state（Streamlit 每次交互都重跑脚本，靠它记住历史）
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# 画出历史对话
for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])
    elif m["role"] == "assistant" and m["content"]:
        # content 为空的是"只调工具不说话"的中间消息，不画
        with st.chat_message("assistant"):
            st.markdown(m["content"])

# 输入框（固定在页面底部）
user_input = st.chat_input("问点什么，比如：什么是 ReAct？")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中（查知识库约需 5~10 秒）..."):
            answer, trace = run_agent(st.session_state.messages)
        st.markdown(answer)

        # 展示这次回答背后的工具调用过程（相当于把 agent_loop.py 的 print 搬上网页）
        if trace:
            with st.expander("🔍 这次回答用到的工具调用"):
                for t in trace:
                    st.code(t, language=None)
