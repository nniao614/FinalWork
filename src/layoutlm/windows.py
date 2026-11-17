"""长票据切窗口后的合并逻辑（docs/algorithm.md 2.2 节）。

本文件只用纯 Python，不依赖 torch / numpy / transformers，方便在 CI 里做单元测试。

背景：
- 超过 512 个子词的票据由分词器切成多个重叠窗口（stride=128）。
- 训练和推理都给分词器传词级标签（推理时传全 0 的假标签），
  分词器只给“第一个子词”打标签，其余子词为 -100。
  **我们用 labels != -100 作为“第一个子词”的标记**，保证训练和推理的规则完全一致。
- transformers 的规则是“与前一个词元属于不同的词就打标签”，遇到特殊符号会重置。
  所以窗口若从一个词的中间开始，开头那个后续子词也会被打上标签：这个词在两个窗口里都有标记。
  它在新窗口里离左边缘的距离是 0，在原窗口里的距离约为 stride，合并时总会选中原窗口，结果正确。

合并规则：一个词出现在多个窗口时，取它离窗口边缘最远的那个窗口。
- 距离 = min(该位置到窗口文本起点的距离, 到窗口文本终点的距离)。
- 文档的开头和结尾不算边缘：第一个窗口的左距离、最后一个窗口的右距离按无穷大处理。
- 距离相同时取较早的窗口。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

IGNORE_INDEX = -100


def first_subword_marks(labels: Sequence[int]) -> List[bool]:
    """分词器生成的标签 → “第一个子词”标记。"""
    return [l != IGNORE_INDEX for l in labels]


def select_word_tokens(
    windows: Sequence[Tuple[Sequence[Optional[int]], Sequence[bool]]],
    n_words: int,
) -> List[Tuple[int, int]]:
    """为一张票据的每个词选出唯一的（窗口号, 词元位置）。

    参数：
        windows: 按顺序排列的窗口，每个元素是 (word_ids, marks)：
            word_ids[t] 为第 t 个词元所属的词下标，特殊符号和填充为 None；
            marks[t] 为该词元是否带标签（labels != -100）。
        n_words: 票据的词数。
    返回：
        长度为 n_words 的列表，第 j 个元素是第 j 个词取值的 (窗口号, 词元位置)。
    若有词在所有窗口里都没有标记，抛出 ValueError（说明切分或标签有问题）。
    """
    n_win = len(windows)
    best: List[Optional[Tuple[float, int, int]]] = [None] * n_words  # (距离, 窗口号, 位置)
    for w, (word_ids, marks) in enumerate(windows):
        if len(word_ids) != len(marks):
            raise ValueError(f"第 {w} 个窗口的 word_ids 与标记长度不一致")
        text_pos = [t for t, j in enumerate(word_ids) if j is not None]
        if not text_pos:
            continue
        start, end = text_pos[0], text_pos[-1]
        for t in text_pos:
            if not marks[t]:
                continue
            j = word_ids[t]
            if not 0 <= j < n_words:
                raise ValueError(f"词下标 {j} 超出范围（共 {n_words} 个词）")
            left = math.inf if w == 0 else t - start
            right = math.inf if w == n_win - 1 else end - t
            dist = min(left, right)
            cur = best[j]
            if cur is None or dist > cur[0]:  # 严格大于：距离相同时保留较早的窗口
                best[j] = (dist, w, t)
    missing = [j for j, b in enumerate(best) if b is None]
    if missing:
        raise ValueError(f"有 {len(missing)} 个词没有取到结果，例如第 {missing[:5]} 个词")
    return [(b[1], b[2]) for b in best]  # type: ignore[index]
