"""阶段3：RAG 最小版 ——「查知识库」核心（不接 Agent）

运行：
    pip install numpy
    python rag_demo.py

观察：程序会对每个问题，从 6 段文档里挑出最相似的 top-2，
     打分越高代表越相关。
     下次课我们再把这一段包装成 Agent 的「查知识库」工具。

前置：.env 里要有 ZHIPU_API_KEY（智谱 embedding-3 是免费的）
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 智谱的 OpenAI 兼容客户端（只用来调 embedding）
# timeout=60：超过 60 秒还没响应就报错退出，避免"假装卡死"
zhipu = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
    timeout=60,
)


# ---------------- 1. 准备知识库（6 段关于 mini-agent 项目的描述） ----------------
DOCS = [
    "项目叫 mini-agent，用纯 Python 从零实现了 Agent，包括工具调用和多步 ReAct 循环",
    "工具调用通过 JSON schema 描述可用函数，模型返回 tool_calls，本地 Python 执行，结果回传",
    "ReAct 是 Agent 的核心循环：思考(Reason) → 行动(Act) → 观察(Observation) → 再思考",
    "本项目在 Windows 上用 Git Bash + Python venv 开发，模型用 DeepSeek，通过 OpenAI 兼容接口调用",
    "工具函数注册到 TOOL_FUNCS 字典，Agent 循环根据模型返回的函数名字符串查表执行",
    "MAX_STEPS 安全阀防止模型无限调用工具，避免烧光 API 余额",
]


# ---------------- 2. 把文档转成向量（embedding） ----------------
def embed(texts):
    """调用智谱 embedding-3，把文字变成 2048 维向量（一串数字，语义的坐标）"""
    resp = zhipu.embeddings.create(model="embedding-3", input=texts)
    return [d.embedding for d in resp.data]


print("正在把 6 段文档转成向量（每次联网请求约 5~10 秒，请耐心等）...")
doc_vectors = np.array(embed(DOCS))   # shape = (6, 2048)
print(f"得到 {doc_vectors.shape[0]} 个向量，每个 {doc_vectors.shape[1]} 维\n")


# ---------------- 3. 检索：问题 → 向量 → 算相似度 → 取 top-k ----------------
def retrieve(query, top_k=2):
    q_vec = np.array(embed([query])[0])
    # 余弦相似度 = 点积 / (||A|| * ||B||)
    # 分数越接近 1，越相关；越接近 0，越不相关
    sims = doc_vectors @ q_vec / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vec)
    )
    top_idx = np.argsort(-sims)[:top_k]   # 相似度从高到低，取前 k 个下标
    return [(DOCS[i], float(sims[i])) for i in top_idx]


# ---------------- 4. 试试几个问题 ----------------
QUERIES = [
    "什么是 ReAct?",                       # 应该 hit "ReAct 是..."
    "工具是怎么被调起来的?",               # 应该 hit "工具调用通过..."
    "项目怎么防止 Agent 无限调用?",        # 应该 hit MAX_STEPS 那段
    "Python 里如何查表执行工具?",          # 相对模糊，观察最坏情况
    "这个 Agent 是用什么模型驱动的?",      # 应该 hit DeepSeek 那段
]

for q in QUERIES:
    print(f"Q: {q}")
    print("  [联网向量化中，约 5~10 秒...]")
    hits = retrieve(q, top_k=2)
    for doc, score in hits:
        print(f"  [{score:.3f}] {doc}")
    print()
