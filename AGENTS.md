# AGENTS.md：给 AI 编程助手的项目说明

本文件写给参与本项目的 AI 助手（Claude、Codex、DeepSeek、Cursor、Copilot 等）。开始任何改动前请先读完。
面向人类读者的介绍见 `README.md`。

**必读顺序**：本文件 → `docs/plan.md`（方案、对照组、实验规则）→ `docs/algorithm.md`（每个模块的算法原理、公式、
数据结构、缓存格式、开发顺序和验收标准）。**写任何模块的代码前，先读 `docs/algorithm.md` 的对应小节，按里面的规格实现。**
规格有不合理或写不通的地方，先停下来告诉用户，不要自行换一种做法。

## 1. 项目是什么

本科毕业设计，暂定题目：**面向票据关键信息抽取的修正收益感知字段级路由方法**。
输入一张票据/发票图像（含 OCR 结果），输出商家、日期、地址、金额等关键字段。

整体流程：

1. **小模型**：LayoutLMv3（`microsoft/layoutlmv3-base`）做词级序列标注，输出每个字段的预测值，以及**票据中所有词**的词级概率和隐藏层向量。
2. **路由网络**（主创新点，自行设计的深度学习模型）：对每个字段预测“交给大模型复核后能被改对”的概率。
   输入为**双路径池化**（片段路径：预测片段内的词做注意力池化；全局路径：每种字段一个可学习查询，对整张票据所有词做注意力池化，
   借鉴导师工作 VST 的 VS-Align；两路门控融合）+ 置信度/熵/完整标签概率向量 + 字段类型/位置 + 一致性特征；
   训练目标为“小模型错且大模型对”（修正收益），而不是“小模型是否出错”。
3. **跨字段一致性约束**（票据特色）：小计 + 税 − 折扣 ≈ 总额、日期/金额格式等，
   既作为路由网络的特征，也用于给大模型生成**定向复核提示**。
4. **大模型**：小参数视觉语言模型（默认 `Qwen/Qwen2.5-VL-3B-Instruct`，LoRA 微调）只复核被选中的字段，
   采用**独立抽取**（提示中不包含小模型的预测值）。
5. **难样本构造**：特征层面模拟小模型错误（借鉴导师工作 PRTR 的在线概率增强 OPA）；
   图像层面拍照场景增强（模糊、阴影、褶皱、透视变形），也作为鲁棒性测试。

论文要证明的是：
- 协同方案**准确率接近“只用大模型”**，**耗时和大模型调用量接近“只用小模型”**；
- 相同调用预算下，路由网络挑出的字段被改对的数量明显多于“按置信度挑”和“随机挑”。

可选扩展（时间充裕时加入，详见 `docs/plan.md` 第 2.5 节）：用大模型修正结果反哺小模型（知识蒸馏）；区域聚焦复核（只截取字段附近区域给大模型）；
用 OCR 结果约束大模型的修正。
写主线代码时请为这些留好接口：大模型复核结果要缓存可复用；复核接口支持传入裁剪后的图像区域；最终结果合并处留一个可开关的校验钩子。

## 2. 当前进度

里程碑 M1～M9 的定义和验收标准见 `docs/algorithm.md` 第 11 节。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 准备 | 项目骨架、统一数据格式、评估指标、配置与工具、测试与 CI、远程开发环境、算法规格文档 | 已完成 |
| 第 1～3 周（11/9～11/29） | M1～M3：数据准备；LayoutLMv3 训练与推理（保存所有词的隐藏层向量与词级 logits）；5 折交叉验证预测；单元构造与错误类型；统一计时与显存测量 | 进行中：M1 已完成（验证集 micro F1：SROIE 0.9344、CORD 0.9758，见 `experiments/m1_layoutlm/summary.md`）；M2、M3 未开始 |
| 第 4～6 周（11/30～12/20） | M4～M5：Qwen2.5-VL 零样本与 LoRA（含 2 折 LoRA）；字段复核接口（独立抽取，支持定向提示）；在交叉验证数据上跑大模型生成路由标签 | 未开始 |
| 第 7～10 周（12/21～1/17） | M6～M8：一致性约束；路由网络训练；特征层面模拟错误；全部对照组与消融实验；错误类型分析 | 未开始 |
| 第 11～13 周（1/18～2/7） | M9：拍照增强、重新 OCR 的端到端测试集、难样本扩充、鲁棒性实验 | 未开始 |
| 之后 | 补充实验、论文、答辩 | 未开始 |

完成一个里程碑后，请更新本表。

## 3. 目录与模块

```
configs/base.yaml       默认配置；实验配置用 `_base_: base.yaml` 继承
data/scripts/           数据准备脚本（prepare_data.py）
data/raw/, processed/   数据（不入库）
src/common.py           读配置、固定种子、选设备、创建实验目录
src/data/               数据集 → 统一格式 Sample（已完成）
  schema.py             Sample 定义、坐标归一化、BIO 工具
  sroie.py              SROIE 解析 + 字段值到词级标签的规则匹配
  cord.py               CORD v2 解析
  io.py                 jsonl 读写、标签表、训练/验证切分、标注率统计
src/eval/               评估：metrics.py（已完成，待补充改对/改错指标）、error_types.py、timing.py、plots.py
src/layoutlm/           LayoutLMv3：windows / dataset / train / infer（M1 已完成）；cv（M2，待写）
src/vlm/                视觉语言模型：prompts / parse / sft / review / cache（待写）
src/router/             路由：units / consistency / simulate / model / train / baselines / pipeline（待写）
src/augment/            拍照场景增强与重新 OCR：photo / reocr（待写）
experiments/            各实验的配置与结果汇总
scripts/check_env.py    环境检查
scripts/run_bg.ps1      Windows 上后台运行长任务
scripts/setup/          Mac / Windows 一键环境配置脚本
docs/plan.md            技术方案、对照组、公平性规则、时间规划
docs/algorithm.md       算法实现规格：原理、公式、数据结构、缓存格式、开发顺序、验收标准
docs/remote-dev.md      远程开发配置说明
tests/                  pytest 单元测试
outputs/                训练输出与缓存（不入库），由 make_run_dir 创建；缓存在 outputs/cache/
```

## 4. 关键数据约定（不要随意改）

所有模块之间只通过 `src/data/schema.py` 里的 `Sample` 交换数据：

```python
Sample(
    id="sroie_train_X00016469612", source="sroie", split="train",
    image_path="raw/sroie/train/img/X00016469612.jpg",  # 相对 data/ 目录，用 / 分隔
    width=463, height=1013,
    words=["TOTAL:", "9.00", ...],          # 词
    boxes=[[x0, y0, x1, y1], ...],          # 原图像素坐标；LayoutLMv3 用 normalized_boxes() 取 0～1000
    labels=["O", "B-TOTAL", ...],           # BIO
    fields={"TOTAL": ["9.00"], ...},        # 字段真实值，值是列表（CORD 一个字段可有多个值）
)
```

- 实体名一律大写。SROIE：`COMPANY / DATE / ADDRESS / TOTAL`；CORD：原类别转大写，如 `MENU.NM`、`TOTAL.TOTAL_PRICE`。
- 标签表 `labels.json` 只从训练集统计，`O` 固定在第 0 位。
- 处理后的数据在 `data/processed/<sroie|cord>/{train,validation,test}.jsonl`。
- SROIE 官方没有验证集，`prepare_data.py` 用固定种子（42）从训练集切出 10% 作为验证集。
- 路由的最小单位是“一张票据的一种字段”（`FieldUnit`，定义见 `docs/algorithm.md` 第 1.1 节）；缓存目录格式见第 1.2 节。

## 5. 实验与评估规则（必须遵守）

- **所有方案用同一套指标**：`src/eval/metrics.py` 的 `field_prf`，在字段值层面比较，预测与真实值都经过 `canonicalize` 归一化。
  小模型的词级预测先用 `decode_bio` → `spans_to_fields` 转成字段值再评估。
- **阈值、温度、早停等一切超参数只能在验证集上调**，测试集只在最终报告时跑一次。
- **对照组必须齐全**（见 `docs/plan.md` 第 3.2 节）：只用小模型、只用大模型、随机路由（同调用比例）、
  置信度阈值路由（未校准/已校准）、整张票据复核、预测“是否出错”的路由网络。
- **路由网络的训练数据必须来自小模型的 5 折交叉验证预测**（out-of-fold），不能用小模型在自己训练数据上的预测。
- **训练集上的大模型结果必须来自 2 折 LoRA**（用一半训练集微调、在另一半上推理），不能用全部训练集微调的 LoRA 在训练集上推理。
  正样本 = 小模型错 ∧ 大模型对。
- **模拟错误样本只用于训练**，验证集和测试集只用真实的交叉验证错误。模拟样本的标签使用不带定向提示的大模型复核缓存结果。
- 所有方法报告：字段级 P/R/F1、单张平均耗时、P95 耗时、吞吐量、显存峰值；协同方案另报大模型调用比例、
  复核后改对与改错的比例、解析失败率（`routing_summary` 目前只算调用比例和耗时，改对/改错比例待按 `docs/algorithm.md` 第 8.1 节补充）。
- **错误类型分析**（`docs/plan.md` 第 3.6 节、`docs/algorithm.md` 第 8.2 节）：每类报告占比、被路由选中比例、被大模型改对比例。
- **计时公平**：同一张显卡；先预热若干张不计入；计时范围一致（包含读图与预处理）；批大小一致；计时实验中大模型真实调用，不读缓存。
- 每次训练/评估用 `make_run_dir` 创建输出目录，自动保存配置和 git 提交号；结果汇总到 `experiments/`。
- 固定随机种子（`set_seed`），对比实验至少报告同一种子下的结果，有时间再补多种子均值。

## 6. 已知问题与设计注意点

1. **SROIE 的词级标签是规则匹配出来的**（官方只给字段值）。改匹配规则后要看 `stats.json` 的 `field_label_rate`，并在论文中说明。
2. **小模型置信度不可靠**：LayoutLMv3 的 softmax 概率通常过于自信，这正是需要路由网络的原因。
   置信度阈值路由作为对照组时，要同时报告未校准和温度缩放校准两个版本。
3. **正样本稀少**：小模型准确率约 95%，“小模型错 ∧ 大模型对”的字段很少。应对：交叉验证预测 + 特征层面模拟错误 + 增强难样本 +
   类别加权或焦点损失。仍不足时，把“是否出错”作为备选训练目标，并作为消融对比。
4. **一致性约束的适用范围**：CORD 有小计、税、总额等字段，可做算术校验；SROIE 只有总额，只能做格式校验。
   **CORD 金额是印尼格式**，`60.000` 表示六万；算术校验用专门的 `parse_amount`，不要用 `canonicalize`。
5. **推理时要保存的中间结果**：LayoutLMv3 推理时保存**票据中所有词**的隐藏层向量和词级 logits（不只是预测片段内的词），
   全局池化路径和模拟错误都依赖它们。路由网络和各种对照路由都离线读取这些结果，不必重复跑小模型。大模型的复核结果同样缓存。
6. **数据增强与 OCR 的关系**：SROIE/CORD 自带的 OCR 文本不会因为图像被模糊、变形而变差。
   如果只增强图像、继续用原始 OCR，鲁棒性实验会失真（透视变形后原始框坐标也对不上）。
   做鲁棒性评估时，应对增强后的图像**重新跑同一个 OCR 引擎**（例如 PaddleOCR），形成端到端测试集。尚未实现。
7. **大模型输出格式**：要求视觉语言模型输出 JSON，解析失败时回退到小模型的结果，并统计解析失败率。
8. **显存**：训练机为 16G 显存。3B 视觉语言模型 + LoRA + bf16 + 梯度检查点可以训练，注意控制图像分辨率（`max_pixels`）。
9. **漏抽字段**：小模型预测为空时没有片段可池化，片段路径用零向量加掩码标记，由全局字段查询路径提供信息。
   漏抽是最需要复核的情况之一，不能在路由时直接跳过。
10. **复核提示不得包含小模型的预测值**（独立抽取），否则大模型结果依赖小模型输出，缓存结果不能复用于模拟错误样本。
    定向提示只描述不一致的类型（如“小计+税与总额不符”），不给出具体的小模型答案。
11. **超长票据**：超过 512 个子词时要切窗口（stride 128），合并时每个词取离窗口边缘最远的那个窗口的结果。
12. **transformers v5 分词器的窗口标签**：窗口从一个词的中间开始时，transformers v5 的分词器会给窗口开头的后续子词也打标签
    （它按“与前一个词元属于不同的词”判断，遇到特殊符号会重置）。推理合并时靠“离边缘距离 + 距离相同取较早窗口”
    选到真正的第一个子词（见 `src/layoutlm/windows.py` 及其单元测试）；训练中会有少量标签噪声，仅涉及超长票据。
    项目把 transformers 固定为 5.22.1，升级前需要重新确认这一行为。

## 7. 环境

- 训练：Windows，RTX 5060 Ti 16G（Blackwell，sm_120），**必须用 CUDA 12.8 版本的 PyTorch**：
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- 辅助：MacBook M1 Max 64G，用 MPS 做数据处理、推理调试和写论文。
- **开发方式**：AI 编程 agent **直接运行在 Windows 台式机上**，工作目录为 `E:\Dev-FinalWork`，虚拟环境为项目下的 `.venv`（uv 管理）。
  用户通过远程方式给 agent 下指令、看结果，不一定在电脑前。（也可以用 Mac 上的 VS Code Remote-SSH 连接，SSH 别名 `win-gpu`，见 `docs/remote-dev.md`。）
  uv 缓存、Python、HuggingFace 模型缓存在 `E:\Dev-cache`（环境变量 `UV_CACHE_DIR`、`UV_PYTHON_INSTALL_DIR`、`HF_HOME`），
  不要把大文件写到 C 盘。
- **长时间任务（训练、批量推理、交叉验证）必须用 `scripts\run_bg.ps1` 后台启动**，日志在 `logs\`：
  `.\scripts\run_bg.ps1 <任务名> "python -m src.layoutlm.train --config configs/layoutlm_sroie.yaml"`。
  agent 启动后**不要在前台等待**，用 `Get-Content logs\<文件名>.log -Tail 50` 查看进度，任务结束后再检查结果。
  正式长任务之前，先用极小的数据量（如 `data.limit=8`、1 个 epoch）试跑一遍，确认能跑通再启动。
- Windows 默认 shell 是 PowerShell，命令要兼容 PowerShell（路径分隔、环境变量写法 `$env:X="..."`）。
- 设备选择用 `src.common.get_device()`，不要在代码里写死 `"cuda"`。
- 路径一律用 `pathlib`，存储的路径用 `/` 分隔，保证 Windows 与 macOS 通用。
- **不要删除或覆盖 `data\`、`outputs\`、`logs\` 下的已有文件**（不在 git 里，删了找不回来）；新结果写进新的运行目录。

## 8. 常用命令

```bash
python scripts/check_env.py                    # 检查环境
python data/scripts/prepare_data.py sroie      # 准备 SROIE
python data/scripts/prepare_data.py cord       # 准备 CORD（国内可设 HF_ENDPOINT=https://hf-mirror.com）
pytest -q                                      # 单元测试
```

## 9. 编码约定

- Python 3.10+，带类型注解。注释、文档字符串、提交信息用**中文**。
- 新模块配套写 `tests/` 下的测试；测试不能依赖真实数据集或 GPU（CI 里没有）。
- 不要提交数据、模型权重、`outputs/`（已在 `.gitignore`）。
- 改动 `Sample` 结构或指标口径前，先说明原因，并同步更新本文件。
- 每完成一个里程碑提交一次；不要一次提交跨多个里程碑的大改动。
- 用户是本科生：改动要附简短解释，思路有问题请直接指出。
