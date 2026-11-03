"""工具注册表：Agent 能用的"手"都在这里

一处定义，多处复用（app.py / agent.py / agent_class.py 都从这里导入），
改工具只改这一个文件，不会出现"网页版和命令行版行为不一致"的问题。
"""

from datetime import datetime

from knowledge import retrieve

# 相似度低于这个值就当作"没检索到"。实测相关问题 top1 多在 0.5 以上，
# 无关问题只有 0.25 左右，0.35 是一条安全分界线——宁可说不知道，也别硬凑。
MIN_SCORE = 0.35


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
    """检索知识库，返回与问题最相关的内容（带来源与相似度）"""
    hits = retrieve(query, top_k=3)
    relevant = [h for h in hits if h["score"] >= MIN_SCORE]
    if not relevant:
        best = hits[0]["score"] if hits else 0.0
        return (
            f"知识库中没有找到相关内容（最高相似度 {best:.2f}，低于阈值 {MIN_SCORE}）。"
            "请直接告诉用户资料不足，不要凭印象回答。"
        )
    return "\n---\n".join(
        f"[{h['source']} · {h['heading']}] 相似度 {h['score']:.2f}\n{h['text']}"
        for h in relevant
    )


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
