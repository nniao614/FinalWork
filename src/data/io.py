"""样本读写：每个划分一个 jsonl 文件，外加 labels.json。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from .schema import Sample, build_label_list, entity_types


def save_jsonl(samples: Iterable[Sample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> List[Sample]:
    with open(path, encoding="utf-8") as f:
        return [Sample.from_dict(json.loads(line)) for line in f if line.strip()]


def save_label_list(samples_by_split: Dict[str, List[Sample]], path: Path) -> List[str]:
    """标签表只从训练集统计，避免测试集里独有的类别混进来。"""
    train = samples_by_split.get("train") or next(iter(samples_by_split.values()))
    ents = set()
    for s in train:
        ents.update(entity_types(s.labels))
    labels = build_label_list(sorted(ents))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    return labels


def dataset_stats(samples: List[Sample]) -> dict:
    """统计每个字段在多少张票据里被成功标注到词上。

    对 SROIE 尤其重要：它的原始标注只给字段值，词级标签是我们自动匹配出来的，
    匹配率太低说明匹配规则需要改。
    """
    n = len(samples)
    if n == 0:
        return {"num_samples": 0}
    has_field = Counter()
    labeled = Counter()
    for s in samples:
        ents = set(entity_types(s.labels))
        for k in s.fields:
            has_field[k] += 1
            if k in ents:
                labeled[k] += 1
    return {
        "num_samples": n,
        "avg_words": round(sum(len(s.words) for s in samples) / n, 1),
        "field_label_rate": {
            k: round(labeled[k] / has_field[k], 3) for k in sorted(has_field)
        },
    }
