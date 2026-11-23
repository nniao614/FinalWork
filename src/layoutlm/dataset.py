"""LayoutLMv3 的输入构造（docs/algorithm.md 2.2 节）。

- 用 `LayoutLMv3Processor`（apply_ocr=False），直接传入 `Sample.words` 和 `Sample.normalized_boxes()`。
- 文本部分用 processor 的分词器一次性编码好（含窗口切分）；图像部分在取数据时用 processor 的
  图像处理器处理并缓存。这与直接调用 processor 的结果相同，只是不必把所有图像同时读进内存，
  且同一张票据的多个窗口只处理一次图像。
- 训练时传真实标签，推理时传全 0 的假标签。分词器只给第一个子词打标签，其余为 -100；
  `labels != -100` 就是“第一个子词”的标记（见 windows.py）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.common import REPO_ROOT
from src.data.io import load_jsonl
from src.data.schema import Sample

from .windows import IGNORE_INDEX, first_subword_marks


@dataclass
class Window:
    """一张票据的一个窗口（不足 512 个子词的票据只有一个窗口）。"""
    sample_idx: int                 # 在样本列表中的下标
    input_ids: List[int]
    attention_mask: List[int]
    bbox: List[List[int]]
    labels: List[int]               # 训练为真实标签；推理为假标签，只用于标记第一个子词
    word_ids: List[Optional[int]]   # 每个词元所属的词下标，特殊符号为 None

    @property
    def marks(self) -> List[bool]:
        return first_subword_marks(self.labels)


# ---------------------------------------------------------------- 数据与标签表

def processed_dir(cfg: Dict[str, Any]) -> Path:
    return REPO_ROOT / cfg["data"]["processed_dir"] / cfg["data"]["dataset"]


def load_split(cfg: Dict[str, Any], split: str) -> List[Sample]:
    """读取一个划分，按 id 排序；配置了 data.limit 时只取前 N 张（快速试跑用）。"""
    samples = sorted(load_jsonl(processed_dir(cfg) / f"{split}.jsonl"), key=lambda s: s.id)
    limit = cfg["data"].get("limit")
    if limit:
        samples = samples[: int(limit)]
    return samples


def load_label_list(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """读取 labels.json（只从训练集统计，O 在第 0 位）。"""
    labels = json.loads((processed_dir(cfg) / "labels.json").read_text(encoding="utf-8"))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    return labels, label2id, id2label


def image_path(cfg: Dict[str, Any], sample: Sample) -> Path:
    return REPO_ROOT / cfg["data"]["image_root"] / sample.image_path


def load_processor(name_or_path: str):
    from transformers import LayoutLMv3Processor
    return LayoutLMv3Processor.from_pretrained(name_or_path, apply_ocr=False)


# ---------------------------------------------------------------- 编码与切窗口

def encode_sample(
    sample: Sample, sample_idx: int, tokenizer, label2id: Optional[Dict[str, int]],
    max_length: int, stride: int,
) -> List[Window]:
    """把一张票据编码成一个或多个窗口。

    label2id 为 None 时表示推理：传全 0 的假标签，只为了得到“第一个子词”的标记。
    """
    if not sample.words:
        raise ValueError(f"{sample.id}: 没有词，无法编码")
    if label2id is None:
        word_labels = [0] * len(sample.words)
    else:
        unknown = sorted({l for l in sample.labels if l not in label2id})
        if unknown:
            raise ValueError(f"{sample.id}: 标签不在标签表中：{unknown}")
        word_labels = [label2id[l] for l in sample.labels]
    enc = tokenizer(
        sample.words, boxes=sample.normalized_boxes(), word_labels=word_labels,
        truncation=True, max_length=max_length, stride=stride,
        return_overflowing_tokens=True, padding=False,
    )
    windows = []
    for w in range(len(enc["input_ids"])):
        windows.append(Window(
            sample_idx=sample_idx,
            input_ids=list(enc["input_ids"][w]),
            attention_mask=list(enc["attention_mask"][w]),
            bbox=[list(b) for b in enc["bbox"][w]],
            labels=list(enc["labels"][w]),
            word_ids=list(enc.word_ids(w)),
        ))
    return windows


def encode_samples(
    samples: List[Sample], tokenizer, label2id: Optional[Dict[str, int]],
    max_length: int, stride: int,
) -> List[Window]:
    out: List[Window] = []
    for i, s in enumerate(samples):
        out.extend(encode_sample(s, i, tokenizer, label2id, max_length, stride))
    return out


def window_stats(samples: List[Sample], windows: List[Window]) -> Dict[str, int]:
    """统计窗口数和超过一个窗口（即超过 512 个子词）的票据数，写进日志。"""
    per_doc: Dict[int, int] = {}
    for w in windows:
        per_doc[w.sample_idx] = per_doc.get(w.sample_idx, 0) + 1
    return {
        "num_samples": len(samples),
        "num_windows": len(windows),
        "num_long_samples": sum(1 for v in per_doc.values() if v > 1),
        "max_windows_per_sample": max(per_doc.values()) if per_doc else 0,
    }


# ---------------------------------------------------------------- Dataset 与拼批

class LayoutLMWindowDataset(Dataset):
    """每个条目是一个窗口；同一张票据的窗口共用一张图（图像处理结果按票据缓存）。"""

    def __init__(self, cfg: Dict[str, Any], samples: List[Sample], windows: List[Window],
                 image_processor, cache_images: bool = True):
        self.cfg = cfg
        self.samples = samples
        self.windows = windows
        self.image_processor = image_processor
        self.cache_images = cache_images
        self._pixels: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def pixel_values(self, sample_idx: int) -> torch.Tensor:
        if sample_idx in self._pixels:
            return self._pixels[sample_idx]
        from PIL import Image
        with Image.open(image_path(self.cfg, self.samples[sample_idx])) as im:
            img = im.convert("RGB")
        pv = self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]
        if self.cache_images:
            self._pixels[sample_idx] = pv
        return pv

    def __getitem__(self, i: int) -> Dict[str, Any]:
        w = self.windows[i]
        return {
            "input_ids": w.input_ids, "attention_mask": w.attention_mask,
            "bbox": w.bbox, "labels": w.labels,
            "pixel_values": self.pixel_values(w.sample_idx),
            "window_idx": i,
        }


def make_collate(pad_token_id: int):
    """按批内最长窗口动态填充（CORD 票据很短，不必都填到 512）。"""

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids, mask, bbox, labels = [], [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_token_id] * pad)
            mask.append(b["attention_mask"] + [0] * pad)
            bbox.append(b["bbox"] + [[0, 0, 0, 0]] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "window_idx": torch.tensor([b["window_idx"] for b in batch], dtype=torch.long),
        }

    return collate
