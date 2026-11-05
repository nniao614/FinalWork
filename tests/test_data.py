"""用模拟样本测试数据转换逻辑（不需要下载真实数据集）。"""

import json
from pathlib import Path

from PIL import Image

from src.data.cord import convert_record
from src.data.io import dataset_stats, load_jsonl, save_jsonl, save_label_list
from src.data.schema import Sample, normalize_box, spans_to_bio
from src.data.sroie import load_sroie, parse_box_file, split_line

BOX_TXT = """﻿72,25,326,25,326,64,72,64,TAN WOON YANN
50,82,440,82,440,121,50,121,BOOK TA .K(TAMAN DAYA) SDN BND
205,121,285,121,285,139,205,139,789417-W
110,144,383,144,383,163,110,163,NO.53 55,57 & 59, JALAN SAGU 18,
192,169,299,169,299,187,192,187,TAMAN DAYA,
162,193,334,193,334,211,162,211,81100 JOHOR BAHRU,
217,216,275,216,275,233,217,233,JOHOR.
50,300,250,300,250,320,50,320,DATE:25/12/2018 8:13:39 PM
50,400,150,400,150,420,50,420,SUBTOTAL
300,400,400,400,400,420,300,420,9.00
50,430,150,430,150,450,50,450,TOTAL:
300,430,400,430,400,450,300,450,9.00
50,460,150,460,150,480,50,480,CASH
300,460,400,460,400,480,300,480,10.00
"""

ENTITIES = {
    "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
    "date": "25/12/2018",
    "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.",
    "total": "9.00",
}


def _make_sroie(root: Path) -> Path:
    base = root / "raw" / "sroie" / "train"
    for d in ("img", "box", "entities"):
        (base / d).mkdir(parents=True)
    Image.new("RGB", (500, 600), "white").save(base / "img" / "X001.jpg")
    (base / "box" / "X001.txt").write_text(BOX_TXT, encoding="utf-8")
    (base / "entities" / "X001.txt").write_text(json.dumps(ENTITIES), encoding="utf-8")
    return root / "raw" / "sroie"


def _spans(sample, entity):
    return [w for w, l in zip(sample.words, sample.labels) if l.endswith("-" + entity)]


def test_parse_box_keeps_commas_and_strips_bom(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text(BOX_TXT, encoding="utf-8")
    lines = parse_box_file(p)
    assert lines[0][0] == "TAN WOON YANN"
    assert lines[3][0] == "NO.53 55,57 & 59, JALAN SAGU 18,"
    assert lines[0][1] == [72, 25, 326, 64]


def test_split_line_boxes_inside_line():
    words = split_line("AB CD", [0, 0, 50, 10])
    assert [w for w, _ in words] == ["AB", "CD"]
    assert words[0][1][0] == 0 and words[1][1][2] == 50


def test_sroie_matching(tmp_path):
    raw = _make_sroie(tmp_path)
    [s] = load_sroie(raw, "train", data_root=tmp_path)
    assert s.image_path == "raw/sroie/train/img/X001.jpg"
    assert (s.width, s.height) == (500, 600)
    # 地址跨 4 行，应全部标上且只有一个 B
    addr = _spans(s, "ADDRESS")
    assert addr[0] == "NO.53" and addr[-1] == "JOHOR."
    assert sum(l == "B-ADDRESS" for l in s.labels) == 1
    # 日期在 "DATE:25/12/2018" 这个词里
    assert _spans(s, "DATE") == ["DATE:25/12/2018"]
    # 9.00 出现两次，应选 TOTAL 那一行，而不是 SUBTOTAL 那一行
    total_idx = [i for i, l in enumerate(s.labels) if l.endswith("TOTAL")]
    assert len(total_idx) == 1 and s.boxes[total_idx[0]][1] == 430
    # 公司名与 OCR 有一个字母差异（BND/BHD），整体覆盖率仍超过 80%，应能兜底匹配
    assert _spans(s, "COMPANY")[0] == "BOOK"
    assert s.fields["TOTAL"] == ["9.00"]


def test_cord_record():
    def quad(x0, y0, x1, y1):
        return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}

    gt = {
        "meta": {"image_size": {"width": 1000, "height": 2000}},
        "valid_line": [
            {"category": "menu.nm", "group_id": 1, "words": [
                {"quad": quad(10, 10, 80, 40), "text": "ICED", "is_key": 0},
                {"quad": quad(90, 10, 160, 40), "text": "TEA", "is_key": 0}]},
            {"category": "total.total_price", "group_id": 2, "words": [
                {"quad": quad(10, 100, 100, 130), "text": "TOTAL", "is_key": 1},
                {"quad": quad(500, 100, 600, 130), "text": "25,000", "is_key": 0}]},
            {"category": "menu.nm", "group_id": 3, "words": [
                {"quad": quad(10, 50, 80, 70), "text": "  ", "is_key": 0}]},
        ],
    }
    # 实际图像缩小一半，坐标应同步缩放
    s = convert_record(gt, "c0", "train", "x.jpg", 500, 1000)
    assert s.words == ["ICED", "TEA", "TOTAL", "25,000"]
    assert s.labels == ["B-MENU.NM", "I-MENU.NM", "O", "B-TOTAL.TOTAL_PRICE"]
    assert s.boxes[0] == [5, 5, 40, 20]
    assert s.fields == {"MENU.NM": ["ICED TEA"], "TOTAL.TOTAL_PRICE": ["25,000"]}

    s2 = convert_record(gt, "c0", "train", "x.jpg", 500, 1000, label_keys=True)
    assert s2.labels[2:] == ["B-TOTAL.TOTAL_PRICE", "I-TOTAL.TOTAL_PRICE"]


def test_roundtrip_labels_stats(tmp_path):
    raw = _make_sroie(tmp_path)
    samples = load_sroie(raw, "train", data_root=tmp_path)
    save_jsonl(samples, tmp_path / "train.jsonl")
    back = load_jsonl(tmp_path / "train.jsonl")
    assert back[0] == samples[0]
    labels = save_label_list({"train": back}, tmp_path / "labels.json")
    assert labels[0] == "O" and "B-TOTAL" in labels and len(labels) == 9
    stats = dataset_stats(back)
    assert stats["field_label_rate"] == {"ADDRESS": 1.0, "COMPANY": 1.0, "DATE": 1.0, "TOTAL": 1.0}


def test_normalize_and_bio():
    assert normalize_box([50, 60, 500, 600], 500, 600) == [100, 100, 1000, 1000]
    assert spans_to_bio(4, [("A", [1, 2]), ("B", [2, 3])]) == ["O", "B-A", "I-A", "B-B"]


def test_length_mismatch_raises():
    try:
        Sample("x", "s", "train", "p", 1, 1, ["a"], [], [])
    except ValueError:
        return
    raise AssertionError("应当报错")
