# Windows 训练机：开启 SSH 远程登录（需要管理员权限，只需运行一次，可重复运行）。
#
# 用法：在“以管理员身份运行”的 PowerShell 里执行
#   powershell -ExecutionPolicy Bypass -File .\setup_windows_admin.ps1 -PublicKey "ssh-ed25519 AAAA... mac-to-win-gpu"
#
# 做的事情：
#   1. 安装并启动 OpenSSH 服务器，设为开机自启，放行防火墙 22 端口
#   2. SSH 登录后的默认 shell 设为 PowerShell
#   3. 写入 Mac 的公钥，实现免密登录
#   4. 用 winget 安装 Git、uv、Tailscale
#   5. 插电时不睡眠（台式机跑训练，睡眠会断开连接、中断训练）

#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory = $true)][string]$PublicKey,
    [switch]$AllowSleep
)
$ErrorActionPreference = "Stop"
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[注意] $m" -ForegroundColor Yellow }

$PublicKey = $PublicKey.Trim()
if ($PublicKey -notmatch '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-\S+) \S+') {
    throw "公钥格式不对，应以 ssh-ed25519 开头。请完整复制 Mac 上 setup_mac.sh 最后输出的那一行。"
}

# ---------- 1. OpenSSH 服务器 ----------
Step "安装 OpenSSH 服务器"
$cap = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*' | Select-Object -First 1
if ($null -eq $cap) { throw "系统里找不到 OpenSSH.Server 组件，请确认是 Windows 10 1809 以上版本" }
if ($cap.State -ne 'Installed') {
    Add-WindowsCapability -Online -Name $cap.Name | Out-Null
    Write-Host "已安装 $($cap.Name)"
} else {
    Write-Host "已安装，跳过"
}
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}
Set-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -Profile Any -Enabled True
Write-Host "防火墙已放行 22 端口（所有网络类型）"

# ---------- 2. 默认 shell ----------
Step "SSH 默认 shell 设为 PowerShell"
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null

# ---------- 3. 公钥 ----------
Step "写入 Mac 公钥"
function Add-Key($file) {
    $dir = Split-Path $file
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    if (-not (Test-Path $file)) { New-Item -ItemType File -Force $file | Out-Null }
    $existing = Get-Content $file -ErrorAction SilentlyContinue
    if ($existing -contains $PublicKey) {
        Write-Host "已存在：$file"
    } else {
        Add-Content -Path $file -Value $PublicKey -Encoding ascii
        Write-Host "已写入：$file"
    }
}
# Windows 的规则：属于管理员组的账户，sshd 只读取 ProgramData 下的这个文件
$adminKeys = "$env:ProgramData\ssh\administrators_authorized_keys"
Add-Key $adminKeys
# 该文件必须只有 Administrators 组和 SYSTEM 有权限，否则 sshd 会拒绝使用（用 SID 避免中文系统组名不同）
icacls $adminKeys /inheritance:r /grant "*S-1-5-32-544:F" /grant "*S-1-5-18:F" | Out-Null
# 普通账户读取用户目录下的文件；两处都写，不管当前账户是不是管理员都能用
Add-Key "$env:USERPROFILE\.ssh\authorized_keys"

Restart-Service sshd

# ---------- 4. 软件 ----------
Step "安装 Git、uv、Tailscale"
if (Get-Command winget -ErrorAction SilentlyContinue) {
    foreach ($id in @("Git.Git", "astral-sh.uv", "tailscale.tailscale")) {
        winget list --id $id -e --accept-source-agreements *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$id 已安装"
        } else {
            winget install --id $id -e --silent --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -ne 0) { Warn "$id 安装失败，请手动安装" }
        }
    }
} else {
    Warn "没有 winget（应用安装程序）。请在微软商店更新“应用安装程序”，或手动安装 Git、uv、Tailscale"
}

# ---------- 5. 电源 ----------
if (-not $AllowSleep) {
    Step "插电时不睡眠、不休眠（可用 -AllowSleep 参数跳过）"
    powercfg /change standby-timeout-ac 0
    powercfg /change hibernate-timeout-ac 0
}

# ---------- 检查 ----------
Step "检查显卡驱动"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} else {
    Warn "找不到 nvidia-smi，请先安装最新的 NVIDIA 显卡驱动"
}

Step "完成。在 Mac 上连接时需要的信息："
Write-Host ("Windows 用户名：" + $env:USERNAME)
Write-Host "局域网 IPv4 地址："
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' } |
    ForEach-Object { Write-Host ("  " + $_.IPAddress + "  (" + $_.InterfaceAlias + ")") }
Write-Host ""
Write-Host "下一步：关掉这个窗口，打开一个普通（非管理员）PowerShell，运行 setup_windows_env.ps1"
Warn "如果 Windows 是用微软账户登录的，SSH 密码登录要用微软账户密码；但配置好公钥后不需要密码"
