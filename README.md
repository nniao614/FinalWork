# 基于大小模型协同的票据关键信息抽取

本科毕业设计。从票据/发票图像中抽取商家、日期、金额等关键字段。

给 AI 编程助手的详细说明（数据约定、评估规则、已知问题）见 [AGENTS.md](AGENTS.md)。

## 方案概述

1. **小模型抽取**：LayoutLMv3 对票据做字段抽取，并为每个字段输出置信度。
2. **置信度路由**（核心创新点）：置信度高于阈值的字段直接采用；低于阈值的字段交给视觉语言模型复核。
3. **大模型复核**：小参数视觉语言模型（Qwen2.5-VL 小版本，LoRA 微调）重新判断低置信度字段。
4. **鲁棒性增强**：针对拍照场景（模糊、阴影、褶皱、透视变形）做数据增强。

## 目录结构

```
configs/            训练、推理、路由的配置文件
data/               数据集（不提交到仓库），data/scripts/ 放下载与预处理脚本
src/data/           SROIE、CORD 转换为统一格式（schema.py 定义样本结构）
src/layoutlm/       LayoutLMv3 训练、推理、置信度输出
src/vlm/            视觉语言模型的 LoRA 微调与复核
src/router/         置信度路由
src/augment/        拍照场景数据增强
src/eval/           指标：准确率、F1、平均耗时、大模型调用比例
experiments/        对比实验与消融实验的配置和结果
scripts/            一键运行脚本
notebooks/          探索性分析
tests/              单元测试（pytest）
```

## 实验设计

| 方案 | 说明 |
|---|---|
| 只用小模型 | LayoutLMv3 单独抽取 |
| 只用大模型 | 视觉语言模型单独抽取 |
| 协同方案 | 置信度路由 |

指标：字段级准确率、F1、单张平均耗时、大模型调用比例。另做消融实验：路由阈值、是否 LoRA 微调、是否加入数据增强。

## 环境

- 训练机：Windows，RTX 5060 Ti 16G（Blackwell 架构，必须使用支持 CUDA 12.8 的 PyTorch）
- 辅助机：MacBook M1 Max（数据处理、推理、写论文，使用 MPS 后端）

Windows 上安装 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/check_env.py   # 检查显卡和依赖是否就绪
```

## 数据集

- SROIE：英文收据，字段为 company、date、address、total
- CORD：印尼收据，字段更细（菜品、单价、小计等）

数据集和模型权重不提交到仓库，见 `.gitignore`。

### 数据准备

```bash
# SROIE：把官方数据解压到 data/raw/sroie/{train,test}/{img,box,entities}
# 官方没有验证集，脚本会用固定种子从训练集切出 10%
python data/scripts/prepare_data.py sroie

# CORD：自动从 HuggingFace 下载；国内网络可先设置镜像
# Windows PowerShell:  $env:HF_ENDPOINT="https://hf-mirror.com"
# macOS:               export HF_ENDPOINT=https://hf-mirror.com
python data/scripts/prepare_data.py cord
```

输出在 `data/processed/<数据集>/`：每个划分一个 jsonl、`labels.json`（标签表）、`stats.json`。

**注意**：SROIE 的原始标注只有字段值，没有词级标签，词级标签是脚本按规则自动匹配的。
运行后务必查看 `stats.json` 里的 `field_label_rate`（每个字段成功标注的比例），
明显低于 0.9 的字段需要抽查样本、调整匹配规则。论文里也应说明这一预处理步骤。

### 测试

```bash
pip install pytest
pytest
```

## 进度

- [ ] 第 1～3 周：读论文，跑通 LayoutLMv3 在 SROIE/CORD 上的基线
- [ ] 第 4～6 周：跑通视觉语言模型抽取及 LoRA 微调
- [ ] 第 7～10 周：实现置信度路由，完成对比实验与消融实验
- [ ] 之后：鲁棒性增强、补充实验、撰写论文
