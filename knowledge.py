"""知识库模块：docs/*.md → 切块 → 向量化（增量缓存）→ 检索

三条设计线：
1. 切块保留标题上下文：块首拼上所属小节标题，避免单句脱离语境后语义漂移
2. 增量缓存：以内容哈希为键，只有新增/改动的块才调 embedding 接口
3. 检索结果带来源：返回文件名、小节、相似度，方便核对原文（也是防幻觉的一层）

被导入时会自动建索引：命中缓存时毫秒级完成，缓存缺失时才走网络。
"""

import hashlib
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
CACHE_FILE = Path(__file__).parent / ".cache" / "kb_vectors.npz"
EMBED_MODEL = "embedding-3"
MAX_CHARS = 420          # 单块字符上限：太大语义被平均，太小上下文不完整
DEFAULT_TOP_K = 3

zhipu = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url="https://open.bigmodel.cn/api/paas/v4",
    timeout=60,
)


# ---------------- 基础能力 ----------------
def _sha(text: str) -> str:
    """内容指纹，用来判断这个块需不需要重新向量化。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """调用智谱 embedding-3，把文字变成 2048 维向量。分批避免单次请求过大。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = zhipu.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


# ---------------- 切块 ----------------
def split_markdown(text: str, source: str, max_chars: int = MAX_CHARS) -> list[dict]:
    """按标题分节，再把段落累加成不超过 max_chars 的块。"""
    chunks: list[dict] = []
    buf: list[str] = []
    heading = ""

    def flush() -> None:
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if body:
            # 块首带上小节标题：单独一句「响应慢可以加缓存」没有语境，
            # 加上标题后向量里才包含「这是关于性能优化的」这层信息
            content = f"【{heading}】\n{body}" if heading else body
            chunks.append(
                {
                    "id": f"{source}#{len(chunks)}",
                    "source": source,
                    "heading": heading,
                    "text": content,
                }
            )
        buf.clear()

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            flush()
            heading = block.lstrip("#").strip()
            continue
        if buf and sum(len(b) for b in buf) + len(block) > max_chars:
            flush()
        buf.append(block)
    flush()
    return chunks


def load_chunks() -> list[dict]:
    """读取 docs/ 下所有 markdown，切块后返回。"""
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        raise RuntimeError(f"知识库为空：{DOCS_DIR} 下没有 .md 文件")
    chunks: list[dict] = []
    for path in files:
        chunks.extend(split_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


# ---------------- 向量缓存 ----------------
def _load_cache() -> dict[str, tuple[str, np.ndarray]]:
    """返回 {块id: (内容哈希, 向量)}；模型变了或文件损坏就当作没有缓存。"""
    if not CACHE_FILE.exists():
        return {}
    try:
        data = np.load(CACHE_FILE, allow_pickle=False)
        if str(data["model"]) != EMBED_MODEL:
            return {}  # 换模型必须整体重建，向量空间不通用
        ids = [str(x) for x in data["ids"]]
        hashes = [str(x) for x in data["hashes"]]
        vectors = data["vectors"]
        return {i: (h, vectors[n]) for n, (i, h) in enumerate(zip(ids, hashes))}
    except Exception:
        return {}  # 缓存只是加速手段，坏了就重建，别让主流程挂掉


def _save_cache(cache: dict[str, tuple[str, np.ndarray]]) -> None:
    if not cache:
        return
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ids = list(cache)
    np.savez(
        CACHE_FILE,
        model=np.array(EMBED_MODEL),
        ids=np.array(ids, dtype=str),
        hashes=np.array([cache[i][0] for i in ids], dtype=str),
        vectors=np.vstack([cache[i][1] for i in ids]).astype(np.float32),
    )


def build_index(force: bool = False) -> tuple[list[dict], np.ndarray]:
    """建索引：只对新增/改动的块调接口，其余直接复用缓存。"""
    chunks = load_chunks()
    cache = {} if force else _load_cache()

    todo = [c for c in chunks if cache.get(c["id"], ("", None))[0] != _sha(c["text"])]
    if todo:
        print(f"[knowledge] {len(todo)}/{len(chunks)} 个块需要向量化，正在请求 embedding-3 ...")
        for chunk, vector in zip(todo, embed([c["text"] for c in todo])):
            cache[chunk["id"]] = (_sha(chunk["text"]), np.asarray(vector, dtype=np.float32))

    # 文档删掉后，缓存里对应的块也要清掉，否则会越积越大
    alive = {c["id"] for c in chunks}
    cache = {k: v for k, v in cache.items() if k in alive}
    _save_cache(cache)

    matrix = np.vstack([cache[c["id"]][1] for c in chunks]).astype(np.float32)
    return chunks, matrix


# ---------------- 模块级索引 ----------------
CHUNKS, MATRIX = build_index()

# 预先归一化：余弦相似度只看方向，归一化之后点积就是余弦值
_NORMS = np.linalg.norm(MATRIX, axis=1, keepdims=True)
_NORMS[_NORMS == 0] = 1.0
UNIT_MATRIX = MATRIX / _NORMS


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """问题 → 向量 → 余弦相似度 → top_k 块（带来源与分数）。"""
    q_vec = np.asarray(embed([query])[0], dtype=np.float32)
    q_norm = np.linalg.norm(q_vec) or 1.0
    sims = UNIT_MATRIX @ (q_vec / q_norm)
    top_idx = np.argsort(-sims)[:top_k]
    return [
        {
            "text": CHUNKS[i]["text"],
            "source": CHUNKS[i]["source"],
            "heading": CHUNKS[i]["heading"],
            "score": float(sims[i]),
        }
        for i in top_idx
    ]


def stats() -> dict:
    """给界面看的知识库概况。"""
    return {
        "files": len({c["source"] for c in CHUNKS}),
        "chunks": len(CHUNKS),
        "dim": int(MATRIX.shape[1]),
        "model": EMBED_MODEL,
    }
