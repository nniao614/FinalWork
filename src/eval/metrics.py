"""字段级评估指标。所有方案（只用小模型 / 只用大模型 / 协同）都用同一套指标，保证可比。

评估粒度是“字段值”而不是词级标签，因为视觉语言模型直接输出字段值，没有词级标签。

流程：
1. 小模型的词级 BIO 预测先用 `decode_bio` 解码成字段值；视觉语言模型直接给字段值。
2. 预测值和真实值都经过 `canonicalize` 归一化（大小写、空格、日期/金额提取）。
3. 每个字段按多重集合匹配统计 TP/FP/FN，算逐字段与总体（micro）P/R/F1。

SROIE 的四个字段每张票据各一个值，此时“字段 F1”与官方任务 3 的口径一致；
CORD 的菜品等字段一张票据有多个值，多重集合匹配可以处理。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

Fields = Dict[str, List[str]]

_DATE_RE = re.compile(
    r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}"                       # 25/12/2018, 2018-12-25
    r"|\d{1,2}\s*[A-Za-z]{3,9}\.?\s*,?\s*\d{2,4}"               # 25 Dec 2018
    r"|[A-Za-z]{3,9}\.?\s*\d{1,2},?\s*\d{2,4}"                 # Dec 25, 2018
)
_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def decode_bio(
    words: Sequence[str], labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
) -> Dict[str, List[dict]]:
    """把词级 BIO 标签解码成实体片段。

    返回 {实体: [{"text": 文本, "idxs": [词下标], "scores": [词级置信度]}, ...]}。
    孤立的 I- 标签视为新片段的开头（模型预测常出现这种情况）。
    """
    out: Dict[str, List[dict]] = {}
    cur: Optional[dict] = None
    cur_ent = None
    for i, lab in enumerate(labels):
        if lab == "O" or lab is None:
            cur, cur_ent = None, None
            continue
        tag, ent = lab.split("-", 1)
        if tag == "B" or ent != cur_ent or cur is None:
            cur = {"words": [], "idxs": [], "scores": []}
            out.setdefault(ent, []).append(cur)
            cur_ent = ent
        cur["words"].append(words[i])
        cur["idxs"].append(i)
        if scores is not None:
            cur["scores"].append(float(scores[i]))
    for spans in out.values():
        for s in spans:
            s["text"] = " ".join(s.pop("words"))
    return out


def spans_to_fields(spans: Dict[str, List[dict]], single_value: Sequence[str] = ()) -> Fields:
    """实体片段转字段值。single_value 里的字段（如 SROIE 的四个字段）只保留一个值：
    选平均置信度最高的片段，没有置信度时选最长的。"""
    fields: Fields = {}
    for ent, ss in spans.items():
        if ent in single_value and len(ss) > 1:
            def key(s):
                sc = s.get("scores") or []
                return (sum(sc) / len(sc) if sc else 0.0, len(s["text"]))
            ss = [max(ss, key=key)]
        fields[ent] = [s["text"] for s in ss]
    return fields


def canonicalize(field: str, value: str) -> str:
    """归一化字段值。预测值和真实值都要过这一步再比较。"""
    v = re.sub(r"\s+", " ", str(value)).strip().upper()
    f = field.upper()
    if f == "DATE" or f.endswith(".DATE"):
        m = _DATE_RE.search(v)
        v = m.group(0) if m else v
        return re.sub(r"\s+", "", v)
    if f == "TOTAL" or "PRICE" in f or f.endswith("CNT"):
        nums = _AMOUNT_RE.findall(v)
        if nums:
            # 金额去掉千分位逗号；票据上的 "RM 9.00"、"TOTAL: 9.00" 都只留数字
            return nums[-1].replace(",", "")
        return v.replace(" ", "")
    # 其他文本字段：去掉标点前后多余空格，例如 ".K (TAMAN" 与 ".K(TAMAN"
    v = re.sub(r"\s*([(),.&:;/-])\s*", r"\1", v)
    return v


def field_prf(preds: Sequence[Fields], golds: Sequence[Fields]) -> dict:
    """逐字段与总体 P/R/F1。preds 和 golds 按票据一一对应。"""
    if len(preds) != len(golds):
        raise ValueError("预测与真实样本数量不一致")
    tp, fp, fn = Counter(), Counter(), Counter()
    doc_correct, doc_total = Counter(), Counter()
    for p, g in zip(preds, golds):
        for f in set(p) | set(g):
            pc = Counter(canonicalize(f, v) for v in p.get(f, []) if str(v).strip())
            gc = Counter(canonicalize(f, v) for v in g.get(f, []) if str(v).strip())
            hit = sum((pc & gc).values())
            tp[f] += hit
            fp[f] += sum(pc.values()) - hit
            fn[f] += sum(gc.values()) - hit
            if gc:
                doc_total[f] += 1
                doc_correct[f] += int(pc == gc)

    def prf(t, p_, n):
        prec = t / (t + p_) if t + p_ else 0.0
        rec = t / (t + n) if t + n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    per_field = {}
    for f in sorted(set(tp) | set(fp) | set(fn)):
        per_field[f] = prf(tp[f], fp[f], fn[f])
        per_field[f]["support"] = tp[f] + fn[f]
        if doc_total[f]:
            per_field[f]["exact_match"] = round(doc_correct[f] / doc_total[f], 4)
    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_f1 = (sum(v["f1"] for v in per_field.values()) / len(per_field)) if per_field else 0.0
    return {"micro": micro, "macro_f1": round(macro_f1, 4), "per_field": per_field}


def routing_summary(n_fields_total: int, n_fields_routed: int, latencies_s: Sequence[float]) -> dict:
    """协同方案特有的指标：大模型调用比例与平均耗时。"""
    lat = list(latencies_s)
    return {
        "vlm_call_ratio": round(n_fields_routed / n_fields_total, 4) if n_fields_total else 0.0,
        "avg_latency_s": round(sum(lat) / len(lat), 4) if lat else 0.0,
        "p95_latency_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 4) if lat else 0.0,
    }
