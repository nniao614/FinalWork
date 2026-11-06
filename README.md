# 面向票据关键信息抽取的修正收益感知字段级路由

本科毕业设计。从票据/发票图像中抽取商家、日期、金额等关键字段，用小模型与大模型协同，
在准确率接近大模型的同时把耗时和大模型调用量降到接近小模型。

- 完整技术方案、对照组与时间规划：[docs/plan.md](docs/plan.md)
- 给 AI 编程助手的说明（数据约定、评估规则、已知问题）：[AGENTS.md](AGENTS.md)

## 方案概述

1. **小模型抽取**：LayoutLMv3 抽取所有字段，输出预测值、词级概率和隐藏层向量。
2. **路由网络**（主创新点）：自行设计的小型神经网络，预测每个字段“交给大模型复核后能否被改对”，
   只把值得复核的字段送给大模型。输入包括小模型的隐藏层向量、置信度、字段类型与位置、一致性校验结果。
3. **跨字段一致性约束**：小计 + 税 − 折扣 ≈ 总额、日期与金额格式等，作为路由特征，并为大模型生成定向复核提示。
4. **大模型复核**：Qwen2.5-VL-3B（LoRA 微调）只复核被选中的字段。
5. **拍照场景增强**：模糊、阴影、褶皱、透视变形，既制造难样本训练路由网络，也用于鲁棒性评估。

## 目录结构

```
configs/            训练、推理、路由的配置文件
data/               数据集（不提交到仓库），data/scripts/ 放下载与预处理脚本
src/data/           SROIE、CORD 转换为统一格式（schema.py 定义样本结构）
src/layoutlm/       LayoutLMv3 训练、推理、置信度输出
src/vlm/            视觉语言模型的 LoRA 微调与复核
src/router/         路由网络、一致性约束、对照路由策略
src/augment/        拍照场景数据增强
src/eval/           指标：准确率、F1、平均耗时、大模型调用比例
experiments/        对比实验与消融实验的配置和结果
scripts/            一键运行脚本
notebooks/          探索性分析
tests/              单元测试（pytest）
```

## 实验设计

| 类别 | 方案 |
|---|---|
| 单模型 | 只用 LayoutLMv3；只用 Qwen2.5-VL（零样本 / LoRA）；可选 Donut |
| 协同对照 | 随机路由（同调用比例）；置信度阈值路由（未校准 / 校准）；整张票据复核；预测“是否出错”的路由网络 |
| 本文 | 修正收益路由网络 + 一致性约束 + 定向提示 |

指标：字段级 P/R/F1、单张平均耗时与 P95、吞吐量、显存峰值、大模型调用比例、复核改对/改错比例。
所有方法使用同一测试集、同一 OCR 输入、同一评估代码、同一张显卡计时。消融实验与公平性规则见 `docs/plan.md`。

## 环境

- 训练机：Windows，RTX 5060 Ti 16G（Blackwell 架构，必须使用支持 CUDA 12.8 的 PyTorch）
- 辅助机：MacBook M1 Max（数据处理、推理、写论文，使用 MPS 后端）

远程开发（Mac 上用 VS Code 连 Windows 训练）的配置步骤见 [docs/remote-dev.md](docs/remote-dev.md)，
包含 Mac 和 Windows 的一键配置脚本（`scripts/setup/`）。

Windows 上手动安装 PyTorch：

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

- [x] 准备：项目骨架、数据格式、评估指标、远程开发环境
- [ ] 第 1～3 周：数据准备；LayoutLMv3 基线；5 折交叉验证预测
- [ ] 第 4～6 周：Qwen2.5-VL 零样本与 LoRA；字段复核接口；生成路由标签
- [ ] 第 7～10 周：一致性约束；路由网络；对照组与消融实验
- [ ] 第 11～13 周：拍照增强与鲁棒性实验
- [ ] 之后：补充实验、论文、答辩
