# 远程开发配置：Mac 写代码，Windows 跑训练

## 整体结构

```
MacBook（VS Code）  ──SSH──▶  Windows 台式机（RTX 5060 Ti）
  编辑代码、看结果               代码、数据、.venv、训练都在这里
  本地也有一份仓库和 .venv        E:\Dev-FinalWork
  （写论文、MPS 推理调试用）
```

- VS Code 的 Remote-SSH 让你在 Mac 上直接编辑 Windows 上的文件、用 Windows 的终端和显卡，体验和本地开发一样。
- 同一个局域网：直接用 Windows 的内网 IP 连接。
- 不在同一个网络（宿舍、实验室、在外面）：两台机器都装 Tailscale 并登录同一个账号，用 Tailscale 分配的地址连接，不需要公网 IP，也不用在路由器上开端口。
- 代码通过 GitHub 同步；两边都有仓库，改完记得提交推送。

## 配置步骤（按顺序做）

### 第 1 步：Mac

打开“终端”，执行：

```bash
curl -fsSL https://raw.githubusercontent.com/nniao614/FinalWork/main/scripts/setup/setup_mac.sh -o /tmp/setup_mac.sh
bash /tmp/setup_mac.sh
```

> 如果仓库已改为私有，上面的地址会 404。先运行 `xcode-select --install` 装好 git，
> 再 `git clone https://github.com/nniao614/FinalWork.git ~/Projects/FinalWork`，然后运行仓库里的 `scripts/setup/setup_mac.sh`。

脚本会：安装 Homebrew、git、uv、VS Code（含 Remote-SSH 等扩展）、Tailscale；生成 SSH 密钥；
克隆仓库到 `~/Projects/FinalWork` 并建好本地 `.venv`。

最后会输出一行以 `ssh-ed25519` 开头的**公钥**，复制下来，下一步要用。

### 第 2 步：Windows（管理员）

1. 先确认已装好最新的 NVIDIA 驱动。
2. 从 GitHub 下载 `scripts/setup/setup_windows_admin.ps1` 和 `setup_windows_env.ps1`（或直接下载仓库 zip）。
3. 开始菜单搜索 PowerShell → 右键“以管理员身份运行”，进入脚本所在目录，执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows_admin.ps1 -PublicKey "粘贴第 1 步的公钥"
```

脚本会：开启 OpenSSH 服务器并设为开机自启、放行防火墙、写入公钥、安装 Git / uv / Tailscale、
设置插电不睡眠（不想要可加 `-AllowSleep`）。最后会显示 **Windows 用户名** 和 **IP 地址**，记下来。

### 第 3 步：Tailscale（不在同一局域网时才需要）

两台机器都打开 Tailscale，用同一个账号登录。在 Tailscale 里能看到两台设备后，
记下 Windows 的机器名或 `100.x.x.x` 地址，后面用它代替内网 IP。

### 第 4 步：Mac 写入连接配置

```bash
bash ~/Projects/FinalWork/scripts/setup/setup_mac.sh <Windows的IP或Tailscale地址> <Windows用户名>
ssh win-gpu        # 测试：不需要密码就能登进 PowerShell 说明成功
```

### 第 5 步：Windows 项目环境

在第 4 步 SSH 登录进去的窗口里（或 Windows 本机的普通 PowerShell）执行：

```powershell
cd E:\   # 第 2 步下载脚本的位置
powershell -ExecutionPolicy Bypass -File .\setup_windows_env.ps1
```

会把缓存目录改到 `E:\Dev-cache`、克隆仓库到 `E:\Dev-FinalWork`，用 uv 建 Python 3.11 的 `.venv`，
安装 CUDA 12.8 版 PyTorch 和项目依赖，最后跑 `check_env.py`。看到显卡型号和 `sm_120` 且没有 [失败] 就对了。

> 第一次 `git clone` 如果仓库是私有的，会弹出 GitHub 登录窗口，这一步要在 Windows 本机上操作一次。

### 第 6 步：VS Code 连上去

1. Mac 上打开 VS Code，按 `Cmd+Shift+P`，输入 **Remote-SSH: Connect to Host**，选 `win-gpu`。
2. 连接成功后 **File → Open Folder**，打开 `E:\Dev-FinalWork`。
3. 按 `Cmd+Shift+P` → **Python: Select Interpreter** → 选 `.venv\Scripts\python.exe`。
4. 远端还需要装一次扩展：左侧扩展栏里，对 Python、Pylance 点“Install in SSH: win-gpu”。

## 日常使用

### 长时间训练一定要用后台方式

Windows 的 SSH 断开时会结束它启动的所有进程。直接在 VS Code 终端里跑训练，Mac 合盖或网络抖一下就白跑了。用：

```powershell
.\scripts\run_bg.ps1 layoutlm_sroie "python -m src.layoutlm.train --config configs/layoutlm_sroie.yaml"
```

- 日志在 `logs\` 下，实时查看：`Get-Content logs\<文件名>.log -Wait -Tail 50`
- 所有后台任务记录在 `logs\jobs.tsv`
- 结束任务：`taskkill /PID <PID> /T /F`
- 看显卡占用：`nvidia-smi -l 5`

几分钟内能跑完的调试，直接在终端里运行即可。

### 看 TensorBoard / Jupyter

在远端启动后，VS Code 会自动把端口转发到 Mac，直接在 Mac 浏览器打开 `http://localhost:6006` 即可。
不用 VS Code 时手动转发：`ssh -N -L 6006:localhost:6006 win-gpu`。

### 传文件

```bash
scp ./某文件 win-gpu:/E:/Dev-FinalWork/data/raw/              # Mac → Windows
scp win-gpu:/E:/Dev-FinalWork/outputs/某实验/metrics.json .    # Windows → Mac
```

数据集比较大，建议直接在 Windows 上下载和预处理，不要从 Mac 传。

## 常见问题

| 现象 | 原因与解决 |
|---|---|
| `Permission denied (publickey)` | 公钥没写对。在 Windows 管理员 PowerShell 重跑第 2 步；确认粘贴的是完整一行 |
| `Connection timed out` | IP 变了或不在同一网络。用 Tailscale 地址；或在路由器里给台式机固定 IP |
| VS Code 一直卡在“正在连接” | 检查 VS Code 设置里有 `"remote.SSH.remotePlatform": {"win-gpu": "windows"}`（第 4 步会自动写入） |
| 终端中文乱码 | 在 PowerShell 执行 `chcp 65001`；或在 Python 里设置 `PYTHONIOENCODING=utf-8` |
| 台式机睡眠后连不上 | 第 2 步已设置插电不睡眠；若之前改过，重跑第 2 步 |
| `torch.cuda.is_available()` 为 False | 驱动太旧或装了非 cu128 的 PyTorch。更新驱动，重跑第 5 步 |
