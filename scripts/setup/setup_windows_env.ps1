# Windows 训练机：克隆项目并创建 Python 环境（普通权限运行，可重复运行）。
#
# 用法：在普通 PowerShell 里执行（在 Windows 本机或从 Mac 用 ssh win-gpu 登录后都可以）
#   powershell -ExecutionPolicy Bypass -File .\setup_windows_env.ps1
#
# 结果：项目在 E:\Dev-FinalWork，虚拟环境在项目下的 .venv（Python 3.11 + CUDA 12.8 版 PyTorch）
#
# 缓存目录：uv 的下载缓存、uv 安装的 Python、HuggingFace 模型默认都在 C 盘，体积十几 GB，
#           这里统一改到 E:\Dev-cache（写入用户环境变量，永久生效）。不想改可传 -CacheRoot ""。

param(
    [string]$ProjDir = "E:\Dev-FinalWork",
    [string]$CacheRoot = "E:\Dev-cache",
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

if ($CacheRoot) {
    Step "缓存目录改到 $CacheRoot"
    $vars = @{
        UV_CACHE_DIR          = "$CacheRoot\uv-cache"
        UV_PYTHON_INSTALL_DIR = "$CacheRoot\uv-python"
        HF_HOME               = "$CacheRoot\huggingface"
    }
    foreach ($k in $vars.Keys) {
        New-Item -ItemType Directory -Force $vars[$k] | Out-Null
        [Environment]::SetEnvironmentVariable($k, $vars[$k], "User")   # 以后打开的窗口生效
        Set-Item -Path "Env:$k" -Value $vars[$k]                       # 当前窗口立即生效
        Write-Host "  $k = $($vars[$k])"
    }
}

Step "获取代码到 $ProjDir"
if (Test-Path "$ProjDir\.git") {
    git -C $ProjDir pull --ff-only
} elseif (-not (Test-Path $ProjDir) -or -not (Get-ChildItem -Force $ProjDir | Select-Object -First 1)) {
    # 文件夹不存在或为空：直接克隆
    git clone $RepoUrl $ProjDir
} else {
    # 文件夹里已有别的文件：在原地初始化仓库，不会删除或覆盖已有文件（有同名文件时 git 会报错停下）
    Write-Host "$ProjDir 不是空文件夹，在原地初始化仓库"
    git -C $ProjDir init -b main
    git -C $ProjDir remote add origin $RepoUrl
    git -C $ProjDir fetch origin
    git -C $ProjDir checkout --track origin/main
    if ($LASTEXITCODE -ne 0) { throw "检出失败：$ProjDir 里有与仓库同名的文件，请先移走后重试" }
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
