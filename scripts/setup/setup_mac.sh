#!/usr/bin/env bash
# Mac 端开发环境一键配置。可重复运行，已装好的部分会跳过。
#
# 用法：
#   bash scripts/setup/setup_mac.sh                     # 只装本机环境、生成 SSH 密钥
#   bash scripts/setup/setup_mac.sh <Windows地址> <Windows用户名>   # 同时写好 SSH 连接配置
#
# Windows 地址：同一局域网填内网 IP（如 192.168.1.23），用 Tailscale 则填 Tailscale 里的机器名或 100.x 地址。
# Windows 用户名：在 Windows 的 PowerShell 里运行 whoami，取反斜杠后面的部分。

set -euo pipefail

WIN_HOST="${1:-}"
WIN_USER="${2:-}"
SSH_ALIAS="win-gpu"
REPO_URL="https://github.com/nniao614/FinalWork.git"
PROJ_DIR="$HOME/Projects/FinalWork"
KEY="$HOME/.ssh/id_ed25519"

step() { printf "\n\033[36m==> %s\033[0m\n" "$1"; }
warn() { printf "\033[33m[注意] %s\033[0m\n" "$1"; }

# ---------- 1. Homebrew ----------
step "检查 Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null \
    || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
fi

# ---------- 2. 命令行工具与应用 ----------
step "安装 git、uv（Python 环境管理）"
brew install git uv

step "安装 VS Code、Tailscale"
for cask in visual-studio-code tailscale; do
  if brew list --cask "$cask" >/dev/null 2>&1; then
    echo "$cask 已安装"
  else
    brew install --cask "$cask" || warn "$cask 安装失败，可能已经手动装过，可忽略"
  fi
done

if command -v code >/dev/null 2>&1; then
  step "安装 VS Code 扩展"
  for ext in ms-vscode-remote.remote-ssh ms-python.python ms-python.vscode-pylance \
             ms-toolsai.jupyter charliermarsh.ruff; do
    code --install-extension "$ext" --force >/dev/null && echo "  $ext"
  done
else
  warn "找不到 code 命令。打开 VS Code，按 Cmd+Shift+P 输入 “Shell Command: Install 'code' command in PATH”，然后重跑本脚本"
fi

# ---------- 3. SSH 密钥与连接配置 ----------
step "SSH 密钥"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -C "mac-to-$SSH_ALIAS" -f "$KEY" -N ""
else
  echo "已存在 $KEY，直接使用"
fi

if [ -n "$WIN_HOST" ] && [ -n "$WIN_USER" ]; then
  step "写入 ~/.ssh/config（别名 $SSH_ALIAS）"
  CFG="$HOME/.ssh/config"
  touch "$CFG" && chmod 600 "$CFG"
  if grep -qE "^Host[[:space:]]+$SSH_ALIAS\$" "$CFG"; then
    warn "$CFG 里已有 Host $SSH_ALIAS，未改动。地址变了请手动编辑该文件"
  else
    cat >> "$CFG" <<EOF

Host $SSH_ALIAS
    HostName $WIN_HOST
    User $WIN_USER
    IdentityFile $KEY
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
EOF
    echo "已添加，之后用 ssh $SSH_ALIAS 连接"
  fi

  step "告诉 VS Code 远端是 Windows"
  VSC_SETTINGS="$HOME/Library/Application Support/Code/User/settings.json"
  mkdir -p "$(dirname "$VSC_SETTINGS")"
  uv run --no-project --python 3.11 python - "$VSC_SETTINGS" "$SSH_ALIAS" <<'PY'
import json, sys, pathlib, re
path, alias = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8") if path.exists() else "{}"
try:
    data = json.loads(text or "{}")
except json.JSONDecodeError:
    # VS Code 设置允许注释，解析失败就不动它
    print(f"settings.json 含注释或格式特殊，请手动加入: \"remote.SSH.remotePlatform\": {{\"{alias}\": \"windows\"}}")
    sys.exit(0)
data.setdefault("remote.SSH.remotePlatform", {})[alias] = "windows"
data.setdefault("remote.SSH.connectTimeout", 30)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("已更新", path)
PY
else
  warn "没有提供 Windows 地址和用户名，跳过 SSH 配置。Windows 那边配好后带参数重跑本脚本即可"
fi

# ---------- 4. 本机项目环境（数据处理、MPS 推理、写论文用） ----------
step "克隆仓库到 $PROJ_DIR"
if [ ! -d "$PROJ_DIR/.git" ]; then
  mkdir -p "$(dirname "$PROJ_DIR")"
  git clone "$REPO_URL" "$PROJ_DIR"
else
  git -C "$PROJ_DIR" pull --ff-only || warn "拉取失败，可能有未提交的改动，已跳过"
fi

step "创建 Python 3.11 虚拟环境（.venv）"
cd "$PROJ_DIR"
[ -d .venv ] || uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch torchvision
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/check_env.py || true

# ---------- 5. 下一步 ----------
step "Mac 端完成。下面是要复制到 Windows 的公钥（整行）："
echo
cat "$KEY.pub"
echo
echo "下一步：在 Windows 上以管理员身份运行 setup_windows_admin.ps1，把上面这行作为 -PublicKey 参数。"
echo "详细步骤见 docs/remote-dev.md"
