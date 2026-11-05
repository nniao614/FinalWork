"""SROIE（ICDAR 2019 任务 3）转换为统一格式。

期望的目录结构（官方发布格式）：

    <root>/train/img/X00016469612.jpg
    <root>/train/box/X00016469612.txt       每行: x1,y1,x2,y2,x3,y3,x4,y4,文本
    <root>/train/entities/X00016469612.txt  JSON: {"company","date","address","total"}
    <root>/test/...                          同上

难点：SROIE 只给出字段的值，没有给出词级标签。这里用规则把值匹配回 OCR 文本行：
- COMPANY / ADDRESS：寻找拼起来恰好等于字段值的连续文本行（忽略空格和大小写）；
  找不到时，在连续 1～6 行里取与字段值相似度最高且不低于 0.85 的一段。
- DATE / TOTAL：在词里查找包含该值的词。TOTAL 常在票据上出现多次，
  优先选同一行或同一水平带里有 "TOTAL" 字样（且不是 SUBTOTAL）的候选，否则取最后一个。
匹配不上的字段不打标签，但字段值仍保留在 `fields` 里，评估时不受影响。
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import Sample, quad_to_xyxy, spans_to_bio

FIELDS = ["company", "date", "address", "total"]
FUZZY_THRESHOLD = 0.85  # 多行字段模糊匹配的最低相似度


# ---------- 文件解析 ----------

def parse_box_file(path: Path) -> List[Tuple[str, List[int]]]:
    """返回 [(行文本, [x0,y0,x1,y1]), ...]，按文件顺序（即大致阅读顺序）。"""
    lines = []
    text = path.read_text(encoding="utf-8", errors="ignore").lstrip("﻿")
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",", 8)  # 文本本身可能含逗号，只切前 8 个
        if len(parts) < 9:
            continue
        try:
            coords = [float(p) for p in parts[:8]]
        except ValueError:
            continue
        content = parts[8].strip()
        if not content:
            continue
        lines.append((content, quad_to_xyxy(coords[0::2], coords[1::2])))
    return lines


def parse_entities(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore").lstrip("﻿").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 少数文件不是严格 JSON，退回到逐个键用正则提取
        data = {}
        for k in FIELDS:
            m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if m:
                data[k] = m.group(1)
    return {k: str(data[k]).strip() for k in FIELDS if data.get(k)}


def split_line(content: str, box: List[int]) -> List[Tuple[str, List[int]]]:
    """把一行拆成词，按字符位置在行框内按比例分配横坐标。"""
    x0, y0, x1, y1 = box
    n = max(len(content), 1)
    out = []
    for m in re.finditer(r"\S+", content):
        wx0 = x0 + (x1 - x0) * m.start() / n
        wx1 = x0 + (x1 - x0) * m.end() / n
        out.append((m.group(), [int(wx0), y0, int(round(wx1)), y1]))
    return out


# ---------- 字段匹配 ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def _match_multiline(value: str, line_texts: List[str]) -> Optional[List[int]]:
    """找拼起来等于 value 的连续行，返回行下标。"""
    target = _norm(value)
    if not target:
        return None
    normed = [_norm(t) for t in line_texts]
    best: Optional[List[int]] = None
    for i, li in enumerate(normed):
        if not li or not target.startswith(li):
            continue
        acc, idxs = li, [i]
        j = i + 1
        while acc != target and j < len(normed):
            if normed[j] and target.startswith(acc + normed[j]):
                acc += normed[j]
                idxs.append(j)
            elif normed[j]:
                break
            j += 1
        if acc == target:
            return idxs

    # 兜底：OCR 文本与标注常有个别字符不同（如 BND / BHD），
    # 在 1～6 行的连续窗口里找相似度最高的一段
    best_ratio = 0.0
    for i in range(len(normed)):
        if not normed[i]:
            continue
        acc = ""
        for j in range(i, min(i + 6, len(normed))):
            acc += normed[j]
            if len(acc) > 1.5 * len(target):
                break
            r = SequenceMatcher(None, acc, target, autojunk=False).ratio()
            if r > best_ratio:
                best_ratio, best = r, [k for k in range(i, j + 1) if normed[k]]
    return best if best_ratio >= FUZZY_THRESHOLD else None


def _same_band(a: List[int], b: List[int]) -> bool:
    """两个框在竖直方向重叠超过一半，视为同一行。"""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    h = min(a[3] - a[1], b[3] - b[1])
    return h > 0 and overlap > 0.5 * h


def _is_total_keyword(text: str) -> bool:
    t = _norm(text)
    return "TOTAL" in t and "SUBTOTAL" not in t


def _match_value_words(
    value: str,
    words: List[str],
    boxes: List[List[int]],
    word_line: List[int],
    line_texts: List[str],
    line_boxes: List[List[int]],
    prefer_total: bool,
) -> Optional[List[int]]:
    target = _norm(value)
    if not target:
        return None
    nw = [_norm(w) for w in words]
    candidates: List[List[int]] = []
    for i in range(len(words)):
        # 从第 i 个词开始拼，直到长度覆盖 target
        acc, idxs = "", []
        for j in range(i, min(i + 6, len(words))):
            acc += nw[j]
            idxs.append(j)
            if target in acc:
                candidates.append(idxs)
                break
            if len(acc) > len(target) + 20:
                break
    # 去掉被更短候选包含的冗长候选
    candidates = [c for c in candidates if not any(set(o) < set(c) for o in candidates)]
    if not candidates:
        return None
    if not prefer_total:
        return candidates[0]

    def near_total(c: List[int]) -> bool:
        li = word_line[c[0]]
        if _is_total_keyword(line_texts[li]):
            return True
        return any(
            _same_band(line_boxes[li], line_boxes[k]) and _is_total_keyword(line_texts[k])
            for k in range(len(line_texts))
        )

    near = [c for c in candidates if near_total(c)]
    return (near or candidates)[-1]


# ---------- 主流程 ----------

def convert_one(
    sid: str, split: str, image_rel: str, width: int, height: int,
    lines: List[Tuple[str, List[int]]], entities: Dict[str, str],
) -> Sample:
    words, boxes, word_line = [], [], []
    line_word_idx: List[List[int]] = []
    for li, (content, box) in enumerate(lines):
        idxs = []
        for w, b in split_line(content, box):
            idxs.append(len(words))
            words.append(w)
            boxes.append(b)
            word_line.append(li)
        line_word_idx.append(idxs)

    line_texts = [c for c, _ in lines]
    line_boxes = [b for _, b in lines]
    spans = []
    for key in ["company", "address"]:
        if key in entities:
            li = _match_multiline(entities[key], line_texts)
            if li:
                spans.append((key.upper(), [w for l in li for w in line_word_idx[l]]))
    for key in ["date", "total"]:
        if key in entities:
            wi = _match_value_words(
                entities[key], words, boxes, word_line, line_texts, line_boxes,
                prefer_total=(key == "total"),
            )
            if wi:
                spans.append((key.upper(), wi))

    return Sample(
        id=sid, source="sroie", split=split, image_path=image_rel,
        width=width, height=height, words=words, boxes=boxes,
        labels=spans_to_bio(len(words), spans),
        fields={k.upper(): [v] for k, v in entities.items()},
    )


def _find_image(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_sroie(root: Path, split: str, data_root: Optional[Path] = None) -> List[Sample]:
    from PIL import Image

    base = root / split
    data_root = data_root or root
    samples = []
    for box_path in sorted((base / "box").glob("*.txt")):
        stem = box_path.stem
        ent_path = base / "entities" / f"{stem}.txt"
        img_path = _find_image(base / "img", stem)
        if img_path is None or not ent_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        samples.append(convert_one(
            sid=f"sroie_{split}_{stem}", split=split,
            image_rel=img_path.relative_to(data_root).as_posix(),
            width=w, height=h,
            lines=parse_box_file(box_path), entities=parse_entities(ent_path),
        ))
    return samples
