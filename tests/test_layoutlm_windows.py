"""长票据窗口切分与合并的单元测试（docs/algorithm.md 2.2 节）。

前几个测试用手工构造的窗口，只依赖纯 Python，CI 可以跑。
最后一个测试用真实分词器，需要 transformers 和本地已下载的模型文件，CI 里会自动跳过。
"""

import pytest

from src.layoutlm.windows import first_subword_marks, select_word_tokens

N = None


def _check_one_result_per_word(sel, n_words):
    assert len(sel) == n_words


def test_first_subword_marks():
    assert first_subword_marks([-100, 3, -100, 0, -100]) == [False, True, False, True, False]


def test_single_window_takes_first_subword():
    # 词 0 有两个子词，词 2 有两个子词
    word_ids = [N, 0, 0, 1, 2, 2, N]
    marks = [False, True, False, True, True, False, False]
    sel = select_word_tokens([(word_ids, marks)], n_words=3)
    assert sel == [(0, 1), (0, 3), (0, 4)]


def test_overlap_picks_window_farthest_from_edge():
    # 窗口 0 的文本位置 1..5，窗口 1 的文本位置 1..4；词 2、3、4 在两个窗口重叠
    w0 = ([N, 0, 1, 2, 3, 4, N], [False, True, True, True, True, True, False])
    w1 = ([N, 2, 3, 4, 5, N], [False, True, True, True, True, False])
    sel = select_word_tokens([w0, w1], n_words=6)
    _check_one_result_per_word(sel, 6)
    # 词 2：窗口 0 距右边缘 2，窗口 1 距左边缘 0 → 窗口 0
    # 词 3：两边距离都是 1 → 取较早的窗口 0
    # 词 4：窗口 0 距右边缘 0，窗口 1 距左边缘 2 → 窗口 1
    assert sel == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4)]


def test_document_end_is_not_an_edge():
    """最后一个窗口的右端是文档结尾，不算边缘（距离按无穷大）。

    词 3 在窗口 0 位于右边缘（距离 0）；在短的最后一个窗口里离左边缘 1、离文档结尾 0。
    若把文档结尾也当边缘，两边距离都是 0，会错误地选窗口 0；正确做法选窗口 1。
    """
    w0 = ([N, 0, 1, 2, 3, N], [False, True, True, True, True, False])
    w1 = ([N, 2, 3, N], [False, True, True, False])
    sel = select_word_tokens([w0, w1], n_words=4)
    assert sel[3] == (1, 2)
    # 词 2：窗口 0 距右边缘 1，窗口 1 距左边缘 0 → 窗口 0
    assert sel[2] == (0, 3)
    # 第一个窗口的左端是文档开头，不算边缘：词 0 的距离为 min(∞, 3)
    assert sel[0] == (0, 1)


def test_window_starting_mid_word():
    """窗口从一个词的中间开始时，transformers 会把开头的后续子词也标为“第一个子词”。
    合并时应选中原窗口里真正的第一个子词。"""
    # 词 2 有 3 个子词；窗口 1 从词 2 的第二个子词开始
    w0 = ([N, 0, 1, 2, 2, 2, 3, N], [False, True, True, True, False, False, True, False])
    w1 = ([N, 2, 2, 3, 4, N], [False, True, False, True, True, False])
    sel = select_word_tokens([w0, w1], n_words=5)
    _check_one_result_per_word(sel, 5)
    assert sel[2] == (0, 3)      # 真正的第一个子词，不是窗口 1 开头的后续子词
    assert sel[3] == (1, 3)      # 窗口 0 距右边缘 0，窗口 1 距左边缘 2
    assert sel[4] == (1, 4)


def test_missing_word_raises():
    w0 = ([N, 0, 1, N], [False, True, False, False])
    with pytest.raises(ValueError):
        select_word_tokens([w0], n_words=2)


def test_real_tokenizer_long_document():
    """用真实分词器构造超过 512 个子词的票据，检查每个词恰好有一个结果，且取到的是真正的第一个子词。"""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("本地没有 layoutlmv3-base 分词器文件")

    words = [f"ITEM{i:04d}xyz" for i in range(300)]
    boxes = [[i % 1000, i % 1000, i % 1000, i % 1000] for i in range(300)]
    enc = tok(words, boxes=boxes, word_labels=[0] * len(words), truncation=True,
              max_length=512, stride=128, return_overflowing_tokens=True)
    n_win = len(enc["input_ids"])
    assert n_win > 1
    windows = [(enc.word_ids(w), first_subword_marks(enc["labels"][w])) for w in range(n_win)]
    sel = select_word_tokens(windows, n_words=len(words))
    _check_one_result_per_word(sel, len(words))

    full = tok(words, boxes=boxes, word_labels=[0] * len(words))
    full_ids = full["input_ids"]
    first_id = {}
    for t, j in enumerate(full.word_ids()):
        if j is not None and j not in first_id:
            first_id[j] = full_ids[t]
    for j, (w, t) in enumerate(sel):
        assert enc.word_ids(w)[t] == j
        assert enc["input_ids"][w][t] == first_id[j]
