"""LayoutLMv3 训练（docs/algorithm.md 2.3 节）。

- 标准词元分类交叉熵（只算第一个子词），AdamW，学习率 5e-5，线性预热 10%，CUDA 上用 bf16。
- 完整模型：在全部训练集上训练，每轮在验证集上算字段级 micro F1（field_prf），
  最多 `layoutlm.epochs` 轮，连续 `early_stop_patience` 轮不提升就停止；记录最佳轮数 E*（从 1 开始）。
- `train_model(..., fixed_epochs=E*)` 固定训练 E* 轮、不看验证集，供 M2 的 5 折交叉验证复用。

输出（运行目录 outputs/<时间>_lm_<数据集>/）：
    best/              最佳权重与 processor（固定轮数时为 final/）
    train_log.json     每轮的训练损失、验证 F1、耗时
    best_epoch.json    E* 与对应的验证 micro F1
    train.log          文本日志

用法：
    python -m src.layoutlm.train --config configs/layoutlm_sroie.yaml
    python -m src.layoutlm.train --config configs/layoutlm_sroie.yaml data.limit=8 layoutlm.epochs=1   # 试跑
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.common import get_device, load_config, make_run_dir, parse_cli, set_seed
from src.data.schema import Sample

from .dataset import (LayoutLMWindowDataset, encode_samples, load_label_list, load_processor,
                      load_split, make_collate, window_stats)
from .infer import autocast_ctx, decode_prediction, evaluate_preds, predict_samples


@dataclass
class TrainResult:
    best_epoch: int                      # E*，从 1 开始；固定轮数训练时等于该轮数
    best_val_f1: Optional[float]
    ckpt_dir: Path
    history: List[Dict[str, Any]] = field(default_factory=list)


def build_model(cfg: Dict[str, Any], labels: List[str]):
    from transformers import LayoutLMv3ForTokenClassification
    return LayoutLMv3ForTokenClassification.from_pretrained(
        cfg["layoutlm"]["model_name"], num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)},
    )


def build_optimizer(model, lm: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW；偏置和 LayerNorm 不做权重衰减（常规做法）。"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith("bias") or "LayerNorm" in n or "norm" in n.lower() else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": lm["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=lm["lr"])


def evaluate(model, processor, samples: List[Sample], cfg: Dict[str, Any], device: str,
             id2label: Dict[int, str], log: Callable[[str], None] = print) -> dict:
    """在一组票据上推理并用 field_prf 评估。"""
    outs = predict_samples(model, processor, samples, cfg, device, log=lambda _m: None)
    single_value = cfg["layoutlm"].get("single_value_fields") or []
    records = [decode_prediction(s, o.logits, id2label, single_value) for s, o in zip(samples, outs)]
    return evaluate_preds(samples, records)


def train_model(
    cfg: Dict[str, Any], train_samples: List[Sample], val_samples: Optional[List[Sample]],
    out_dir: Path, fixed_epochs: Optional[int] = None, log: Callable[[str], None] = print,
) -> TrainResult:
    """训练 LayoutLMv3。

    val_samples 不为空且 fixed_epochs 为 None：按验证集 micro F1 早停，保存最佳权重到 out_dir/best。
    fixed_epochs 给定：固定训练这么多轮、不看验证集，保存到 out_dir/final（M2 交叉验证用）。
    """
    from transformers import get_linear_schedule_with_warmup

    lm = cfg["layoutlm"]
    device = get_device(cfg["device"])
    labels, label2id, id2label = load_label_list(cfg)
    processor = load_processor(lm["model_name"])
    model = build_model(cfg, labels).to(device)

    windows = encode_samples(train_samples, processor.tokenizer, label2id, lm["max_length"], lm["stride"])
    log(f"设备 {device}；训练窗口统计：{window_stats(train_samples, windows)}")
    ds = LayoutLMWindowDataset(cfg, train_samples, windows, processor.image_processor)
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(ds, batch_size=lm["batch_size"], shuffle=True, generator=gen,
                        num_workers=lm["num_workers"], collate_fn=make_collate(processor.tokenizer.pad_token_id))

    use_early_stop = fixed_epochs is None and bool(val_samples)
    max_epochs = fixed_epochs if fixed_epochs is not None else lm["epochs"]
    accum = lm["grad_accum"]
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * max_epochs
    optimizer = build_optimizer(model, lm)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(lm["warmup_ratio"] * total_steps), total_steps)
    ckpt_dir = out_dir / ("best" if use_early_stop else "final")

    history: List[Dict[str, Any]] = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, start=1):
            batch.pop("window_idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device, lm["bf16"]):
                loss = model(**batch).loss
            (loss / accum).backward()
            total_loss += loss.item()
            n_batches += 1
            if step % accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        rec: Dict[str, Any] = {"epoch": epoch, "train_loss": round(total_loss / max(n_batches, 1), 5),
                               "lr": scheduler.get_last_lr()[0], "train_time_s": round(time.time() - t0, 1)}
        if use_early_stop:
            t1 = time.time()
            metrics = evaluate(model, processor, val_samples, cfg, device, id2label)
            f1 = metrics["micro"]["f1"]
            rec.update({"val_micro_f1": f1, "val_macro_f1": metrics["macro_f1"],
                        "eval_time_s": round(time.time() - t1, 1)})
            if f1 > best_f1:
                best_f1, best_epoch, bad_epochs = f1, epoch, 0
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                rec["saved"] = True
            else:
                bad_epochs += 1
        history.append(rec)
        log(json.dumps(rec, ensure_ascii=False))
        (out_dir / "train_log.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if use_early_stop and bad_epochs >= lm["early_stop_patience"]:
            log(f"验证集 F1 连续 {bad_epochs} 轮没有提升，在第 {epoch} 轮停止")
            break

    if not use_early_stop:
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        best_epoch = max_epochs
    return TrainResult(best_epoch=best_epoch, best_val_f1=best_f1 if use_early_stop else None,
                       ckpt_dir=ckpt_dir, history=history)


def main(argv: Optional[list] = None) -> None:
    cfg_path, overrides = parse_cli(argv)
    cfg = load_config(cfg_path, overrides)
    set_seed(cfg["seed"])
    dataset = cfg["data"]["dataset"]
    run_dir = make_run_dir(cfg, f"lm_{dataset}" + ("_smoke" if cfg["data"].get("limit") else ""))
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"运行目录：{run_dir}")
    train_samples = load_split(cfg, "train")
    val_samples = load_split(cfg, "validation")
    log(f"训练 {len(train_samples)} 张，验证 {len(val_samples)} 张")
    result = train_model(cfg, train_samples, val_samples, run_dir, log=log)
    info = {"best_epoch": result.best_epoch, "best_val_micro_f1": result.best_val_f1,
            "ckpt_dir": str(result.ckpt_dir.relative_to(run_dir.parent.parent)).replace("\\", "/")}
    (run_dir / "best_epoch.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成：E* = {result.best_epoch}，最佳验证 micro F1 = {result.best_val_f1}，权重 {result.ckpt_dir}")
    log_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
