"""检查训练环境是否就绪。在 Windows 训练机和 Mac 上各跑一次。

    python scripts/check_env.py

重点检查：RTX 5060 Ti 是 Blackwell 架构（计算能力 sm_120），
只有用 CUDA 12.8 及以上编译的 PyTorch 才能在它上面运行。
"""

from __future__ import annotations

import importlib
import platform
import sys

OK, WARN, FAIL = "[通过]", "[注意]", "[失败]"


def check_pkg(name: str) -> None:
    try:
        mod = importlib.import_module(name)
        print(f"{OK} {name} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"{FAIL} {name} 未安装或导入失败：{e}")


def main() -> None:
    print(f"Python {sys.version.split()[0]}，系统 {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 10):
        print(f"{WARN} 建议使用 Python 3.10～3.12")

    try:
        import torch
    except ImportError:
        print(f"{FAIL} 未安装 PyTorch，见 README 的安装命令")
        return
    print(f"{OK} torch {torch.__version__}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{OK} CUDA {torch.version.cuda}，显卡 {name}，计算能力 sm_{cap[0]}{cap[1]}，显存 {mem:.1f} GB")
        arch_list = torch.cuda.get_arch_list()
        if cap[0] >= 12 and not any(a.startswith("sm_12") for a in arch_list):
            print(f"{FAIL} 当前 PyTorch 不支持 sm_{cap[0]}{cap[1]}，请安装 cu128 版本")
        try:
            x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            (x @ x).sum().item()
            print(f"{OK} GPU bf16 矩阵乘法正常")
        except Exception as e:  # noqa: BLE001
            print(f"{FAIL} GPU 计算出错：{e}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        x = torch.randn(64, 64, device="mps")
        (x @ x).sum().item()
        print(f"{OK} Apple MPS 可用（适合推理和小规模调试，正式训练请用 Windows 机器）")
    else:
        print(f"{WARN} 没有可用的 GPU，只能用 CPU")

    for name in ["transformers", "datasets", "peft", "accelerate", "seqeval", "PIL", "cv2", "yaml"]:
        check_pkg(name)


if __name__ == "__main__":
    main()
