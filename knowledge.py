"""知识库模块：文档 + 向量化 + 检索

这个文件专门给 agent_loop.py 导入用（from knowledge import retrieve），
里面只有"能力"，没有演示代码——这是"模块"和"脚本"的区别。
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 智谱的 OpenAI 兼容客户端（只用来调 embedding）
zhipu = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
    timeout=60,
)


# ---------------- 知识库：6 段关于 mini-agent 项目的描述 ----------------
DOCS = [
    "项目叫 mini-agent，用纯 Python 从零实现了 Agent，包括工具调用和多步 ReAct 循环",
    "工具调用通过 JSON schema 描述可用函数，模型返回 tool_calls，本地 Python 执行，结果回传",
    "ReAct 是 Agent 的核心循环：思考(Reason) → 行动(Act) → 观察(Observation) → 再思考",
    "本项目在 Windows 上用 Git Bash + Python venv 开发，模型用 DeepSeek，通过 OpenAI 兼容接口调用",
    "工具函数注册到 TOOL_FUNCS 字典，Agent 循环根据模型返回的函数名字符串查表执行",
    "MAX_STEPS 安全阀防止模型无限调用工具，避免烧光 API 余额",
]


# ---------------- 向量化 ----------------
def embed(texts):
    """调用智谱 embedding-3，把文字变成 2048 维向量"""
    resp = zhipu.embeddings.create(model="embedding-3", input=texts)
    return [d.embedding for d in resp.data]


# 被导入时就把文档向量化好（只跑一次，之后所有检索共用这份向量）
print("正在把知识库文档转成向量（约 5~10 秒）...")
doc_vectors = np.array(embed(DOCS))   # shape = (6, 2048)
print("知识库就绪，开始对话\n")


# ---------------- 检索 ----------------
def retrieve(query, top_k=2):
    """问题 → 向量 → 余弦相似度 → 返回最相关的 top_k 段"""
    q_vec = np.array(embed([query])[0])
    sims = doc_vectors @ q_vec / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec)
    )
    top_idx = np.argsort(-sims)[:top_k]
    return [(DOCS[i], float(sims[i])) for i in top_idx]
