"""工具注册表：Agent 能用的"手"都在这里

一处定义，多处复用（app.py / agent.py / agent_class.py 都从这里导入），
改工具只改这一个文件，不会出现"网页版和命令行版行为不一致"的问题。
"""

from datetime import datetime

from knowledge import retrieve


# ---------- 工具函数本体 ----------
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


# ---------- 给模型看的说明书（JSON Schema） ----------
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


# ---------- 模型返回的函数名字符串 → 真实函数 ----------
TOOL_FUNCS = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_docs": search_docs,
}
