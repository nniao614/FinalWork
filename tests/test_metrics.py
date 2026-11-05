from src.eval.metrics import (
    canonicalize, decode_bio, field_prf, routing_summary, spans_to_fields)


def test_decode_bio_handles_stray_i():
    words = ["A", "B", "C", "D", "E"]
    labels = ["B-X", "I-X", "O", "I-X", "B-Y"]
    spans = decode_bio(words, labels, scores=[0.9, 0.8, 0.1, 0.5, 0.7])
    assert [s["text"] for s in spans["X"]] == ["A B", "D"]
    assert spans["X"][0]["idxs"] == [0, 1] and spans["X"][0]["scores"] == [0.9, 0.8]
    assert spans["Y"][0]["text"] == "E"


def test_spans_to_fields_single_value_picks_most_confident():
    spans = decode_bio(["9.00", "x", "10.00"], ["B-TOTAL", "O", "B-TOTAL"], [0.6, 0, 0.95])
    assert spans_to_fields(spans, single_value=["TOTAL"]) == {"TOTAL": ["10.00"]}
    assert spans_to_fields(spans) == {"TOTAL": ["9.00", "10.00"]}


def test_canonicalize():
    assert canonicalize("DATE", "DATE:25/12/2018") == "25/12/2018"
    assert canonicalize("TOTAL", "RM 1,234.50") == "1234.50"
    assert canonicalize("TOTAL", "TOTAL: 9.00") == "9.00"
    assert canonicalize("COMPANY", "book ta .k(taman daya)  sdn bhd") == \
        canonicalize("COMPANY", "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert canonicalize("TOTAL.TOTAL_PRICE", "25,000") == "25000"


def test_field_prf():
    golds = [{"TOTAL": ["9.00"], "DATE": ["25/12/2018"]},
             {"TOTAL": ["5.00"], "DATE": ["01/01/2019"]}]
    preds = [{"TOTAL": ["RM9.00"], "DATE": ["DATE:25/12/2018"]},
             {"TOTAL": ["6.00"]}]
    r = field_prf(preds, golds)
    assert r["per_field"]["TOTAL"]["f1"] == 0.5
    assert r["per_field"]["DATE"] == {
        "precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2, "exact_match": 0.5}
    assert r["micro"]["precision"] == 0.6667 and r["micro"]["recall"] == 0.5


def test_field_prf_multiset():
    golds = [{"MENU.NM": ["TEA", "TEA", "CAKE"]}]
    preds = [{"MENU.NM": ["TEA", "CAKE", "BREAD"]}]
    r = field_prf(preds, golds)["per_field"]["MENU.NM"]
    assert (r["precision"], r["recall"]) == (0.6667, 0.6667)


def test_routing_summary():
    r = routing_summary(100, 15, [0.1, 0.2, 0.3, 1.0])
    assert r["vlm_call_ratio"] == 0.15 and r["avg_latency_s"] == 0.4
