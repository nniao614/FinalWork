"""把 SROIE / CORD 转成统一格式。

用法（在仓库根目录运行）：

    # SROIE：先把官方数据解压到 data/raw/sroie/{train,test}/{img,box,entities}
    python data/scripts/prepare_data.py sroie

    # CORD：自动从 HuggingFace 下载（国内网络可设置 HF_ENDPOINT=https://hf-mirror.com）
    python data/scripts/prepare_data.py cord

输出：
    data/processed/<数据集>/{train,validation,test}.jsonl
    data/processed/<数据集>/labels.json
    data/processed/<数据集>/stats.json
所有 image_path 都相对于 data/ 目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.io import dataset_stats, save_jsonl, save_label_list  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["sroie", "cord"])
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--raw", type=Path, default=None, help="SROIE 原始数据目录")
    ap.add_argument("--hf-name", default=None, help="CORD 的 HF 名称或本地目录")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--label-keys", action="store_true",
                    help="CORD：把 is_key 的提示词也标成实体（默认标 O）")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = data_dir / "processed" / args.dataset
    by_split = {}

    if args.dataset == "sroie":
        from src.data.sroie import load_sroie
        raw = (args.raw or data_dir / "raw" / "sroie").resolve()
        for split in ["train", "test"]:
            if (raw / split / "box").exists():
                by_split[split] = load_sroie(raw, split, data_root=data_dir)
        if not by_split:
            sys.exit(f"在 {raw} 下没有找到 train/box 或 test/box，请检查目录结构")
    else:
        from src.data.cord import HF_NAME, load_cord
        for split in ["train", "validation", "test"]:
            by_split[split] = load_cord(
                split, img_dir=out_dir / "images", data_root=data_dir,
                hf_name=args.hf_name or HF_NAME, cache_dir=args.cache_dir,
                label_keys=args.label_keys,
            )

    stats = {}
    for split, samples in by_split.items():
        n = save_jsonl(samples, out_dir / f"{split}.jsonl")
        stats[split] = dataset_stats(samples)
        print(f"[{args.dataset}] {split}: {n} 条")
    labels = save_label_list(by_split, out_dir / "labels.json")
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"标签数 {len(labels)}，统计写入 {out_dir / 'stats.json'}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
