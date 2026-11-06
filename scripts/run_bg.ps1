# 在 Windows 上后台运行长时间任务（训练、评估），SSH 断开或 Mac 合盖后任务不会中断。
#
# 原因：Windows 的 OpenSSH 会在连接断开时结束该连接启动的所有进程，
#       直接在 VS Code 终端里跑训练，网络一抖就前功尽弃。
#       本脚本通过 WMI 创建进程，使其脱离 SSH 会话独立运行。
#
# 用法（在项目根目录）：
#   .\scripts\run_bg.ps1 layoutlm_sroie "python -m src.layoutlm.train --config configs/layoutlm_sroie.yaml"
#   （整条命令必须用引号括起来，否则 -m、-v 之类会被 PowerShell 当成本脚本的参数）
#
# 查看日志：  Get-Content logs\<文件名>.log -Wait -Tail 50
# 查看状态：  Get-Process python
# 结束任务：  taskkill /PID <脚本输出的 PID> /T /F

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Name,
    [Parameter(Mandatory = $true, Position = 1)][string]$Command
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "${stamp}_${Name}.log"

$inner = $Command.Trim()
if ($Name -notmatch '^[\w\-.]+$') { throw "任务名只能包含字母、数字、下划线、横线和点" }

$activate = Join-Path $root ".venv\Scripts\activate.bat"
if (-not (Test-Path $activate)) { throw "找不到 $activate，请先运行 scripts\setup\setup_windows_env.ps1" }

# 无缓冲输出 + UTF-8，日志才能实时看到且中文不乱码
$cmdLine = 'cmd.exe /c "set "PYTHONUNBUFFERED=1" && set "PYTHONIOENCODING=utf-8" && call "' + $activate +
           '" && ' + $inner + ' > "' + $log + '" 2>&1"'

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmdLine; CurrentDirectory = $root }
if ($r.ReturnValue -ne 0) { throw "启动失败，错误码 $($r.ReturnValue)" }

"$($r.ProcessId)`t$stamp`t$Name`t$inner" | Add-Content (Join-Path $logDir "jobs.tsv") -Encoding utf8
Write-Host "已在后台启动，PID $($r.ProcessId)" -ForegroundColor Green
Write-Host "日志：$log"
Write-Host "实时查看：Get-Content `"$log`" -Wait -Tail 50"
Write-Host "结束任务：taskkill /PID $($r.ProcessId) /T /F"
