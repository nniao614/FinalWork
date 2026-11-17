"""LayoutLMv3 推理与缓存（docs/algorithm.md 1.2、2.5 节）。

对每张票据保存：
- `<sample_id>.npz`：所有词的最后一层隐藏层向量 `hidden`（float16，[n_words, 768]）和
  `logits`（float32，[n_words, n_labels]），都取每个词第一个子词的值；交叉验证时另有 `fold`；
- `preds.jsonl`：用 decode_bio + select_spans 得到的字段值和片段词下标。

缓存目录：outputs/cache/layoutlm/<dataset>/<run_tag>/<split>/，run_tag 为训练运行目录名。
训练集用完整模型推理的结果写在 `train_full/`，只用于 M2 的检查，不得用于训练路由网络。

用法（在验证集、测试集、训练集上推理；只对验证集和训练集计算 F1，测试集只写缓存）：
    python -m src.layoutlm.infer --config configs/layoutlm_sroie.yaml \
        layoutlm.ckpt=outputs/<训练运行目录>/best infer.splits=[validation,test,train]
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import REPO_ROOT, get_device, git_commit, load_config, make_run_dir, parse_cli, set_seed
from src.data.schema import Sample
from src.eval.metrics import decode_bio, field_prf, select_spans

from .dataset import LayoutLMWindowDataset, encode_samples, make_collate, window_stats
from .windows import select_word_tokens

# 划分名 → 缓存目录名。训练集用完整模型推理的结果叫 train_full，与 M2 的 oof_train 区分。
CACHE_SPLIT_NAME = {"train": "train_full", "validation": "validation", "test": "test"}
# 只写缓存、不计算 F1 的划分（测试集只在最终报告时评估一次）
NO_EVAL_SPLITS = {"test"}


@dataclass
class WordOutputs:
    """一张票据所有词的小模型输出（每个词取第一个子词的值）。"""
    hidden: np.ndarray   # float16, [n_words, hidden_size]
    logits: np.ndarray   # float32, [n_words, n_labels]


def autocast_ctx(device: str, bf16: bool):
    """只在 CUDA 上启用 bf16 自动混合精度；MPS / CPU 用 float32。"""
    if bf16 and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def forward_hidden_logits(model, batch: Dict[str, torch.Tensor]):
    """前向计算，返回文本部分的最后一层隐藏层向量和 logits。

    与 LayoutLMv3ForTokenClassification.forward 的计算完全相同（取文本部分 → dropout → 分类层），
    只是把分类层之前的隐藏层向量也拿出来；推理时 model.eval()，dropout 不起作用。
    """
    out = model.layoutlmv3(
        input_ids=batch["input_ids"], bbox=batch["bbox"],
        attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"],
    )
    seq_len = batch["input_ids"].shape[1]
    hidden = out[0][:, :seq_len]           # 去掉后面的图像块
    logits = model.classifier(model.dropout(hidden))
    return hidden, logits


@torch.no_grad()
def predict_samples(
    model, processor, samples: List[Sample], cfg: Dict[str, Any], device: str,
    log=print,
) -> List[WordOutputs]:
    """对一组票据推理，返回每张票据所有词的 hidden 和 logits（长票据已做窗口合并）。"""
    lm = cfg["layoutlm"]
    model.eval()
    windows = encode_samples(samples, processor.tokenizer, None, lm["max_length"], lm["stride"])
    log(f"推理窗口统计：{window_stats(samples, windows)}")

    # 先按窗口合并规则确定每个词取哪个窗口的哪个位置，推理时只收集这些位置
    by_sample: Dict[int, List[int]] = {}
    for g, w in enumerate(windows):
        by_sample.setdefault(w.sample_idx, []).append(g)
    wanted: Dict[int, List[tuple]] = {}   # 全局窗口号 → [(词下标, 位置)]
    for si, gs in by_sample.items():
        sel = select_word_tokens([(windows[g].word_ids, windows[g].marks) for g in gs],
                                 n_words=len(samples[si].words))
        for j, (w_local, t) in enumerate(sel):
            wanted.setdefault(gs[w_local], []).append((j, t))

    hidden_size = model.config.hidden_size
    n_labels = model.config.num_labels
    outs = [WordOutputs(np.zeros((len(s.words), hidden_size), dtype=np.float16),
                        np.zeros((len(s.words), n_labels), dtype=np.float32)) for s in samples]

    ds = LayoutLMWindowDataset(cfg, samples, windows, processor.image_processor, cache_images=False)
    loader = DataLoader(ds, batch_size=lm["eval_batch_size"], shuffle=False,
                        num_workers=lm["num_workers"], collate_fn=make_collate(processor.tokenizer.pad_token_id))
    for batch in loader:
        win_idx = batch.pop("window_idx").tolist()
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device, lm["bf16"]):
            hidden, logits = forward_hidden_logits(model, batch)
        hidden = hidden.float().cpu().numpy()
        logits = logits.float().cpu().numpy()
        for b, g in enumerate(win_idx):
            si = windows[g].sample_idx
            for j, t in wanted.get(g, []):
                outs[si].hidden[j] = hidden[b, t].astype(np.float16)
                outs[si].logits[j] = logits[b, t]
    return outs


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def decode_prediction(
    sample: Sample, logits: np.ndarray, id2label: Dict[int, str], single_value: Sequence[str],
) -> Dict[str, Any]:
    """词级 logits → 字段值与片段词下标。

    词级置信度取预测标签的 softmax 概率（单值字段选片段时用）。
    返回 {"id", "fields": {字段: [值]}, "spans": {字段: [[词下标], ...]},
          "raw_span_counts": {字段: 选择前的片段数}}。
    """
    probs = softmax(logits.astype(np.float64))
    pred = probs.argmax(axis=-1)
    conf = probs[np.arange(len(pred)), pred]
    labels = [id2label[int(p)] for p in pred]
    spans = decode_bio(sample.words, labels, conf.tolist())
    chosen = select_spans(spans, single_value)
    return {
        "id": sample.id,
        "fields": {f: [s["text"] for s in ss] for f, ss in chosen.items()},
        "spans": {f: [s["idxs"] for s in ss] for f, ss in chosen.items()},
        "raw_span_counts": {f: len(ss) for f, ss in spans.items()},
    }


def evaluate_preds(samples: List[Sample], records: List[Dict[str, Any]]) -> dict:
    """用统一的 field_prf 计算字段级 P/R/F1。"""
    return field_prf([r["fields"] for r in records], [s.fields for s in samples])


def multi_span_stats(records: List[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    """单值字段预测出多个片段（选择之前）的票据比例。"""
    n = len(records)
    out: Dict[str, Any] = {}
    for f in fields:
        k = sum(1 for r in records if r["raw_span_counts"].get(f, 0) > 1)
        out[f] = {"num_docs": k, "ratio": round(k / n, 4) if n else 0.0}
    return out


# ---------------------------------------------------------------- 缓存

def cache_dir_for(cfg: Dict[str, Any], run_tag: str, split: str) -> Path:
    return (REPO_ROOT / cfg["cache_dir"] / "layoutlm" / cfg["data"]["dataset"] / run_tag
            / CACHE_SPLIT_NAME.get(split, split))


def write_cache(out_dir: Path, sample_id: str, out: WordOutputs, fold: Optional[int] = None) -> None:
    arrays = {"hidden": out.hidden.astype(np.float16), "logits": out.logits.astype(np.float32)}
    if fold is not None:
        arrays["fold"] = np.int64(fold)
    np.savez(out_dir / f"{sample_id}.npz", **arrays)


def write_preds_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_cached(out_dir: Path, sample: Sample) -> WordOutputs:
    """读取一张票据的缓存，并检查词数是否与样本一致。"""
    with np.load(out_dir / f"{sample.id}.npz") as z:
        hidden, logits = z["hidden"], z["logits"]
    if hidden.shape[0] != len(sample.words) or logits.shape[0] != len(sample.words):
        raise ValueError(f"{sample.id}: 缓存词数 {hidden.shape[0]} 与样本词数 {len(sample.words)} 不一致")
    return WordOutputs(hidden, logits)


def write_split_cache(
    out_dir: Path, samples: List[Sample], outs: List[WordOutputs], records: List[Dict[str, Any]],
    meta: Dict[str, Any], folds: Optional[List[int]] = None,
) -> None:
    """写一个划分的全部缓存。目录已存在且非空时报错，不覆盖已有结果。"""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"缓存目录已存在且非空，不覆盖：{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (s, o) in enumerate(zip(samples, outs)):
        write_cache(out_dir, s.id, o, None if folds is None else folds[i])
    write_preds_jsonl(out_dir / "preds.jsonl", records)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 入口

def main(argv: Optional[list] = None) -> None:
    from transformers import LayoutLMv3ForTokenClassification

    from .dataset import load_label_list, load_processor, load_split

    cfg_path, overrides = parse_cli(argv)
    cfg = load_config(cfg_path, overrides)
    set_seed(cfg["seed"])
    lm = cfg["layoutlm"]
    if not lm.get("ckpt"):
        raise ValueError("请用 layoutlm.ckpt=<训练运行目录>/best 指定权重")
    ckpt = Path(lm["ckpt"])
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    run_tag = ckpt.parent.name
    splits = (cfg.get("infer") or {}).get("splits") or ["validation", "test"]
    dataset = cfg["data"]["dataset"]

    run_dir = make_run_dir(cfg, f"lm_infer_{dataset}" + ("_smoke" if cfg["data"].get("limit") else ""))
    log_file = open(run_dir / "infer.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    device = get_device(cfg["device"])
    labels, _, id2label = load_label_list(cfg)
    processor = load_processor(str(ckpt))
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(ckpt)).to(device)
    if model.config.num_labels != len(labels):
        raise ValueError("权重的标签数与 labels.json 不一致")
    log(f"设备 {device}，权重 {ckpt}，划分 {splits}")

    single_value = lm.get("single_value_fields") or []
    summary: Dict[str, Any] = {"run_tag": run_tag, "ckpt": str(ckpt)}
    for split in splits:
        samples = load_split(cfg, split)
        outs = predict_samples(model, processor, samples, cfg, device, log=log)
        records = [decode_prediction(s, o.logits, id2label, single_value) for s, o in zip(samples, outs)]
        out_dir = cache_dir_for(cfg, run_tag, split)
        meta = {
            "dataset": dataset, "split": split, "cache_split": CACHE_SPLIT_NAME.get(split, split),
            "run_tag": run_tag, "ckpt": str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/")
            if ckpt.is_relative_to(REPO_ROOT) else str(ckpt),
            "labels": labels, "hidden_dtype": "float16", "logits_dtype": "float32",
            "num_samples": len(samples), "limit": cfg["data"].get("limit"),
            "single_value_fields": single_value, "git_commit": git_commit(),
        }
        if split == "train":
            meta["note"] = "完整模型在训练集上的预测，只用于 M2 的检查（OOF F1 应低于此 F1），不得用于训练路由网络。"
        write_split_cache(out_dir, samples, outs, records, meta)
        log(f"[{split}] 缓存已写入 {out_dir}")
        if split in NO_EVAL_SPLITS:
            log(f"[{split}] 只写缓存，不计算 F1")
            continue
        metrics = evaluate_preds(samples, records)
        result = {"metrics": metrics}
        if single_value:
            result["multi_span"] = multi_span_stats(records, single_value)
        summary[split] = result
        log(f"[{split}] micro F1 = {metrics['micro']['f1']}")
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"结果已写入 {run_dir / 'metrics.json'}")
    log_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
