# Windows 训练机：克隆项目并创建 Python 环境（普通权限运行，可重复运行）。
#
# 用法：在普通 PowerShell 里执行（在 Windows 本机或从 Mac 用 ssh win-gpu 登录后都可以）
#   powershell -ExecutionPolicy Bypass -File .\setup_windows_env.ps1
#
# 结果：项目在 %USERPROFILE%\Projects\FinalWork，虚拟环境在项目下的 .venv（Python 3.11 + CUDA 12.8 版 PyTorch）

param(
    [string]$ProjDir = "$env:USERPROFILE\Projects\FinalWork",
    [string]$RepoUrl = "https://github.com/nniao614/FinalWork.git"
)
$ErrorActionPreference = "Stop"
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

# 刚用 winget 装完的软件，当前窗口的 PATH 里还没有，从注册表重新读一次
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")
foreach ($c in @("git", "uv")) {
    if (-not (Get-Command $c -ErrorAction SilentlyContinue)) {
        throw "找不到 $c。请先以管理员身份运行 setup_windows_admin.ps1，或手动安装后重开 PowerShell"
    }
}

Step "克隆仓库到 $ProjDir"
if (-not (Test-Path "$ProjDir\.git")) {
    New-Item -ItemType Directory -Force (Split-Path $ProjDir) | Out-Null
    git clone $RepoUrl $ProjDir
} else {
    git -C $ProjDir pull --ff-only
}
Set-Location $ProjDir
# 换行符交给仓库里的 .gitattributes 统一管理
git config core.autocrlf false

Step "创建 Python 3.11 虚拟环境（.venv）"
if (-not (Test-Path ".venv")) { uv venv .venv --python 3.11 }
$py = ".venv\Scripts\python.exe"

Step "安装 CUDA 12.8 版 PyTorch（5060 Ti 必须用这个版本）"
uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128

Step "安装项目依赖"
uv pip install --python $py -r requirements.txt

Step "检查环境"
& $py scripts\check_env.py

Write-Host "`n完成。之后在 VS Code 里打开 $ProjDir，Python 解释器选择 .venv\Scripts\python.exe" -ForegroundColor Green
