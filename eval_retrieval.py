#!/usr/bin/env python3
"""检索质量评测：固定问题集 → 命中率 / MRR / 无关问题拒绝率

用法：
    python eval_retrieval.py            # 表格 + 汇总
    python eval_retrieval.py --json     # 机器可读，方便对比改动前后
    python eval_retrieval.py --top-k 5  # 看更深的召回深度（默认 5）

四个指标：
    Hit@1     top1 命中期望文档的比例 —— 用户看到的第一条对不对
    Hit@3     期望文档进前 3 的比例 —— 召回够不够用
    MRR       正确文档排名倒数的均值 —— 排得越靠前越高
    可回答率   top1 分数达到 MIN_SCORE 的比例 —— 会不会被阈值误杀
拒绝率      无关问题中没有任何块过阈值的比例 —— 会不会硬凑答案

为什么四个一起看：把阈值调高能让无关问题全被拒，但相关问题也会被误杀。
命中与拒绝是一对拉扯，只报一个数字都是自欺欺人。

问题措辞故意不照抄文档原句——照抄的话，检索靠关键词重合就能过，测不出语义能力。
"""

import argparse
import json
import sys
import unicodedata

from knowledge import retrieve, stats
from tools import MIN_SCORE

# (问题, 期望命中的文档, 期望命中的小节)
IN_SCOPE: list[tuple[str, str, str]] = [
    # 01 项目概览
    ("这个项目一共用了几个第三方依赖包？", "01-项目概览.md", "技术栈"),
    ("代码分别放在哪几个文件里，各自负责什么？", "01-项目概览.md", "目录结构"),
    ("对话模型和向量模型分别用的是哪一家的？", "01-项目概览.md", "技术栈"),
    # 02 ReAct 循环
    ("模型是怎么判断下一步该去调工具的？", "02-react循环.md", "核心思想"),
    ("为什么带 tool_calls 的 assistant 消息要原样放回去？", "02-react循环.md", "两个容易踩的坑"),
    ("模型卡在同一个工具上反复调用停不下来，代码里怎么拦？", "02-react循环.md", "为什么要 max_steps"),
    # 03 工具调用
    ("想新增一个工具，需要改动哪些地方？", "03-工具调用.md", "注册表设计"),
    ("工具描述字段写得含糊会有什么后果？", "03-工具调用.md", "协议本身"),
    ("工具执行的时候报错了，对话会不会直接中断？", "03-工具调用.md", "异常处理"),
    # 04 RAG 实现
    ("一篇长文档为什么不整个拿去向量化？", "04-rag实现.md", "文档切块"),
    ("只改了一篇资料，重新启动还要等很久吗？", "04-rag实现.md", "向量化与缓存"),
    ("算余弦相似度之前为什么要把向量归一化？", "04-rag实现.md", "检索"),
    # 05 部署与配置
    ("为什么不能直接用 python 跑主程序？", "05-部署与配置.md", "本地启动"),
    ("本地和线上的密钥是怎么用同一份代码兼容的？", "05-部署与配置.md", "API Key 管理"),
    ("部署到云端之后，向量缓存会一直保留吗？", "05-部署与配置.md", "云端部署"),
    # 06 常见问题
    ("模型会不会瞎编答案，代码里做了哪些防线？", "06-常见问题.md", "怎么防止模型胡说"),
    ("聊到几十轮之后 token 成本会一直涨吗？", "06-常见问题.md", "上下文会无限增长吗"),
    ("摘要生成失败了，整轮对话会崩掉吗？", "06-常见问题.md", "摘要失败会怎样"),
    ("检索的相似度阈值是怎么定下来的？", "06-常见问题.md", "相似度阈值是怎么定的"),
]

# 知识库完全覆盖不到的问题：期望全部被阈值挡住
OUT_OF_SCOPE: list[str] = [
    "明天上海的天气怎么样",
    "帮我写一首关于春天的诗",
    "PyTorch 怎么把模型放到 GPU 上训练",
    "红烧肉的家常做法",
    "明年世界杯的冠军会是谁",
]


# ---------------- 终端对齐（中文占两格） ----------------
def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _cut(text: str, width: int) -> str:
    """按显示宽度截断，超出部分用省略号，避免表格被长问题撑歪。"""
    if _width(text) <= width:
        return text
    out, used = "", 0
    for ch in text:
        used += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used > width - 1:
            break
        out += ch
    return out + "…"


# ---------------- 评测主体 ----------------
def evaluate(top_k: int = 5) -> dict:
    rows: list[dict] = []
    for question, expect_source, expect_heading in IN_SCOPE:
        hits = retrieve(question, top_k=top_k)
        rank = next(
            (i for i, h in enumerate(hits, 1) if h["source"] == expect_source), None
        )
        top1 = hits[0]
        rows.append(
            {
                "question": question,
                "expect": expect_source,
                "rank": rank,
                "top1_source": top1["source"],
                "top1_heading": top1["heading"],
                "top1_score": round(top1["score"], 3),
                "section_ok": top1["heading"] == expect_heading,
                "answered": top1["score"] >= MIN_SCORE,
            }
        )

    oos: list[dict] = []
    for question in OUT_OF_SCOPE:
        top1 = retrieve(question, top_k=top_k)[0]
        oos.append(
            {
                "question": question,
                "best_score": round(top1["score"], 3),
                "best_source": top1["source"],
                "rejected": top1["score"] < MIN_SCORE,
            }
        )

    total = len(rows)
    return {
        "kb": stats(),
        "min_score": MIN_SCORE,
        "top_k": top_k,
        "rows": rows,
        "out_of_scope": oos,
        "metrics": {
            "hit@1": round(sum(r["rank"] == 1 for r in rows) / total, 4),
            "hit@3": round(sum(r["rank"] is not None and r["rank"] <= 3 for r in rows) / total, 4),
            "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / total, 4),
            "answered": round(sum(r["answered"] for r in rows) / total, 4),
            "section@1": round(sum(r["section_ok"] for r in rows) / total, 4),
            "reject": round(sum(o["rejected"] for o in oos) / len(oos), 4),
        },
        "margin": {
            "worst_relevant": min(r["top1_score"] for r in rows),
            "best_irrelevant": max(o["best_score"] for o in oos),
        },
    }


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def report_human(result: dict) -> str:
    rows, oos, m = result["rows"], result["out_of_scope"], result["metrics"]
    kb = result["kb"]
    lines: list[str] = []

    lines.append(
        f"知识库：{kb['files']} 篇文档 · {kb['chunks']} 个块 · {kb['dim']} 维 · "
        f"阈值 {result['min_score']} · top_k={result['top_k']}"
    )
    lines.append("")

    head = (
        _pad("序", 3)
        + _pad("问题", 36)
        + _pad("期望文档", 20)
        + _pad("top1 文档", 20)
        + _pad("分数", 6)
        + _pad("名次", 5)
        + "结论"
    )
    lines.append(head)
    lines.append("-" * (_width(head) + 2))

    for i, row in enumerate(rows, 1):
        ok = row["rank"] == 1
        verdict = "命中" if ok else ("前3" if row["rank"] else "未召回")
        if not row["answered"]:
            verdict += "/低于阈值"
        lines.append(
            _pad(str(i), 3)
            + _pad(_cut(row["question"], 35), 36)
            + _pad(row["expect"], 20)
            + _pad(row["top1_source"], 20)
            + _pad(f"{row['top1_score']:.2f}", 6)
            + _pad(str(row["rank"] or "-"), 5)
            + verdict
        )

    lines.append("")
    lines.append("无关问题（期望全部被拒绝）")
    for row in oos:
        mark = "已拒绝" if row["rejected"] else f"误放行 → {row['best_source']}"
        lines.append(f"  {_pad(_cut(row['question'], 34), 34)}最高分 {row['best_score']:.2f}  {mark}")

    lines.append("")
    lines.append("─" * 60)
    lines.append(f"文档级 Hit@1    {_pct(m['hit@1'])}   （{round(m['hit@1'] * len(rows))}/{len(rows)}，用户看到的第一条）")
    lines.append(f"文档级 Hit@3    {_pct(m['hit@3'])}   （{round(m['hit@3'] * len(rows))}/{len(rows)}，召回够不够）")
    lines.append(f"小节级 Hit@1    {_pct(m['section@1'])}   （命中文档且命中预期小节）")
    lines.append(f"MRR             {m['mrr']:.3f}   （1.0 表示全部排第一）")
    lines.append(f"可回答率        {_pct(m['answered'])}   （top1 分数达到阈值，不会被误杀）")
    lines.append(f"无关问题拒绝率  {_pct(m['reject'])}   （{sum(o['rejected'] for o in oos)}/{len(oos)}）")

    low, high = result["margin"]["worst_relevant"], result["margin"]["best_irrelevant"]
    gap = low - high
    lines.append("")
    if high < result["min_score"] <= low and gap >= 0.05:
        lines.append(
            f"阈值 {result['min_score']} 落在 [{high:.2f}, {low:.2f}] 空档内（宽 {gap:.2f}）："
            f"相关问题最低 {low:.2f}，无关问题最高 {high:.2f}，边界清晰。"
        )
    elif high < result["min_score"] <= low:
        lines.append(
            f"阈值 {result['min_score']} 可区分当前样本，但空档只有 {gap:.2f} 宽"
            f"（相关问题最低 {low:.2f}，无关问题最高 {high:.2f}）。"
            "边界很窄，语料或提问风格一变就可能误判——补更多评测样本再确认，别把这当成稳定结论。"
        )
    else:
        lines.append(
            f"⚠ 阈值 {result['min_score']} 不在空档内：相关问题最低 {low:.2f}，"
            f"无关问题最高 {high:.2f}，存在误判风险，需要调整阈值或补充文档。"
        )

    missed = [r for r in rows if r["rank"] != 1]
    if missed:
        lines.append("")
        lines.append("未拿到 top1 的用例（排查用）：")
        for row in missed:
            lines.append(
                f"  · {row['question']}\n"
                f"    期望 {row['expect']} / 实际 {row['top1_source']} · {row['top1_heading']}"
                f"（{row['top1_score']:.2f}，名次 {row['rank'] or '未召回'}）"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非表格")
    parser.add_argument("--top-k", type=int, default=5, help="召回深度，默认 5")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RATIO",
        help="Hit@1 低于该值时以退出码 1 结束（可用于 CI）",
    )
    args = parser.parse_args()

    result = evaluate(top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(report_human(result))

    if args.fail_under is not None and result["metrics"]["hit@1"] < args.fail_under:
        print(
            f"\nHit@1 {result['metrics']['hit@1']:.3f} 低于门槛 {args.fail_under}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
