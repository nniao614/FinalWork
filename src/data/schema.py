"""统一的票据样本格式。

SROIE 和 CORD 都会被转换成 `Sample`，后续的 LayoutLMv3、视觉语言模型、
路由和评估模块只依赖这一种格式。

约定：
- words / boxes / labels 三个列表等长，一个元素对应一个词。
- boxes 存原图像素坐标 [x0, y0, x1, y1]；喂给 LayoutLMv3 时再用
  `normalized_boxes()` 换算到 0～1000。
- labels 使用 BIO 标注，例如 "B-TOTAL"、"I-ADDRESS"、"O"。
- fields 是字段的真实值，用于字段级评估和视觉语言模型的监督。
  一个字段可能出现多次（如 CORD 的菜品名），所以值是列表。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class Sample:
    id: str
    source: str            # "sroie" 或 "cord"
    split: str             # "train" / "validation" / "test"
    image_path: str        # 相对于数据根目录的路径
    width: int
    height: int
    words: List[str] = field(default_factory=list)
    boxes: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    fields: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (len(self.words) == len(self.boxes) == len(self.labels)):
            raise ValueError(
                f"{self.id}: words/boxes/labels 长度不一致 "
                f"({len(self.words)}/{len(self.boxes)}/{len(self.labels)})"
            )

    def normalized_boxes(self) -> List[List[int]]:
        """换算到 LayoutLM 系列要求的 0～1000 坐标。"""
        return [normalize_box(b, self.width, self.height) for b in self.boxes]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**d)


def normalize_box(box: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box

    def clip(v: float) -> int:
        return max(0, min(1000, int(round(v))))

    return [
        clip(1000 * x0 / width),
        clip(1000 * y0 / height),
        clip(1000 * x1 / width),
        clip(1000 * y1 / height),
    ]


def quad_to_xyxy(xs: List[float], ys: List[float]) -> List[int]:
    """四边形（可能倾斜）转外接矩形。"""
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def spans_to_bio(n_words: int, spans: List[tuple]) -> List[str]:
    """spans: [(entity, [word_idx, ...]), ...]，每个 span 的词按阅读顺序排列。

    同一个词若被多个 span 覆盖，以先出现的 span 为准。
    """
    labels = ["O"] * n_words
    for entity, idxs in spans:
        first = True
        for i in idxs:
            if labels[i] != "O":
                continue
            labels[i] = ("B-" if first else "I-") + entity
            first = False
    return labels


def entity_types(labels: List[str]) -> List[str]:
    return sorted({l[2:] for l in labels if l != "O"})


def build_label_list(entities: List[str]) -> List[str]:
    """生成模型用的标签表：O 在第 0 位，之后每个实体一对 B/I。"""
    out = ["O"]
    for e in sorted(entities):
        out += [f"B-{e}", f"I-{e}"]
    return out
