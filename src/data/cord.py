"""CORD v2 转换为统一格式。

数据来源：HuggingFace `naver-clova-ix/cord-v2`（train 800 / validation 100 / test 100）。
每条记录有 `image`（PIL 图像）和 `ground_truth`（JSON 字符串），其中：

    gt["valid_line"] = [
        {"category": "menu.nm", "group_id": 3,
         "words": [{"quad": {"x1":..,"y1":..,...,"y4":..}, "text": "ICED", "is_key": 0}, ...]},
        ...
    ]

处理规则：
- 一个 valid_line 是一个实体实例，第一个词标 B-，其余标 I-。
- 类别名转成大写作为实体名，如 "menu.nm" -> "MENU.NM"。
- is_key=1 的词是提示性文字（如 "TOTAL"、"CASH"），默认标为 O，也不计入字段值，
  这样模型学的是“值”而不是“标题”。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schema import Sample, quad_to_xyxy, spans_to_bio

HF_NAME = "naver-clova-ix/cord-v2"


def convert_record(
    gt: dict, sid: str, split: str, image_rel: str, width: int, height: int,
    label_keys: bool = False,
) -> Sample:
    # 按 CORD 原始尺寸记录的坐标，若与实际图像尺寸不同需要缩放
    meta_size = (gt.get("meta") or {}).get("image_size") or {}
    sx = width / meta_size["width"] if meta_size.get("width") else 1.0
    sy = height / meta_size["height"] if meta_size.get("height") else 1.0

    words, boxes, spans = [], [], []
    fields: Dict[str, List[str]] = {}
    for line in gt.get("valid_line", []):
        entity = line["category"].upper()
        idxs, value_words = [], []
        for w in line.get("words", []):
            text = (w.get("text") or "").strip()
            if not text:
                continue
            q = w["quad"]
            box = quad_to_xyxy(
                [q["x1"] * sx, q["x2"] * sx, q["x3"] * sx, q["x4"] * sx],
                [q["y1"] * sy, q["y2"] * sy, q["y3"] * sy, q["y4"] * sy],
            )
            i = len(words)
            words.append(text)
            boxes.append(box)
            if w.get("is_key", 0) and not label_keys:
                continue
            idxs.append(i)
            value_words.append(text)
        if idxs:
            spans.append((entity, idxs))
            fields.setdefault(entity, []).append(" ".join(value_words))

    return Sample(
        id=sid, source="cord", split=split, image_path=image_rel,
        width=width, height=height, words=words, boxes=boxes,
        labels=spans_to_bio(len(words), spans), fields=fields,
    )


def load_cord(
    split: str, img_dir: Path, data_root: Path, hf_name: str = HF_NAME,
    cache_dir: Optional[str] = None, label_keys: bool = False,
) -> List[Sample]:
    """从 HuggingFace 读取并把图像落盘到 img_dir；image_path 记录为相对 data_root 的路径。

    hf_name 也可以是本地目录（例如手动下载的 parquet 文件夹）。
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, rec in enumerate(ds):
        sid = f"cord_{split}_{i:04d}"
        image = rec["image"].convert("RGB")
        img_path = img_dir / f"{sid}.jpg"
        if not img_path.exists():
            image.save(img_path, quality=95)
        gt = json.loads(rec["ground_truth"])
        samples.append(convert_record(
            gt, sid=sid, split=split,
            image_rel=img_path.relative_to(data_root).as_posix(),
            width=image.width, height=image.height, label_keys=label_keys,
        ))
    return samples
