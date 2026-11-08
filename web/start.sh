#!/usr/bin/env bash
# 启动看板 + 四个可视化页面（棋谱回放 / 对弈 / MCTS / 网络前向）
#
# Usage:
#   ./web/start.sh                          # 前台跑（调试用，Ctrl-C 停）
#   ./web/start.sh --port 8770 --daemon     # 后台常驻（推荐）
#   ./web/start.sh --stop [--port 8770]     # 停掉后台实例
#
# ⚠️ 必须用 .venv 里的 python：engine.py 要 import torch/numpy，
#    系统 python3 没装这两个，会直接 ImportError（症状：看板好的、对弈页坏的）。
#
# ⚠️ 为什么 --daemon 用 start_new_session=True 而不是普通 nohup &：
#    macOS 没有 setsid 命令，普通后台任务仍留在启动者的进程组里，
#    宿主会话清理/退出时会连带把整组杀掉。症状 = 看板用着好好的，
#    过一会儿变成「连接被拒」，而 web.log 里只有 ConnectionResetError
#    （那是浏览器先断的噪声，不是死因）。独立会话后才真正脱离。
set -u
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "❌ 找不到 .venv/bin/python（对弈/MCTS/前向页需要 torch）" >&2
  exit 1
fi

PORT=8770
MODE=fg
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --stop)   MODE=stop; shift ;;
    --daemon) MODE=daemon; shift ;;
    --port)   PORT="$2"; ARGS+=("--port" "$2"); shift 2 ;;
    *)        ARGS+=("$1"); shift ;;
  esac
done

PIDFILE="logs/web.$PORT.pid"
LOGFILE="logs/web.$PORT.log"
mkdir -p logs

running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local p
  p=$(cat "$PIDFILE" 2>/dev/null)
  [ -n "$p" ] || return 1
  kill -0 "$p" 2>/dev/null || return 1
  # 必须是本目录的 web/server.py，别拿旧 pid 杀到别的进程
  ps -o command= -p "$p" 2>/dev/null | grep -q "web/server.py" || return 1
  echo "$p"
}

case "$MODE" in
  stop)
    if p=$(running_pid); then
      kill "$p" 2>/dev/null
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$p" 2>/dev/null || break
        sleep 0.3
      done
      kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
      echo "已停止 :$PORT (pid $p)"
    else
      echo ":${PORT} 没有在跑"
    fi
    rm -f "$PIDFILE"
    exit 0
    ;;

  daemon)
    if p=$(running_pid); then
      echo ":$PORT 已经在跑 (pid $p) → http://127.0.0.1:$PORT/"
      exit 0
    fi
    rm -f "$PIDFILE"
    # start_new_session=True == setsid，脱离当前会话/进程组
    NEWPID=$(.venv/bin/python - "$PORT" "$LOGFILE" "${ARGS[@]:-}" <<'PY'
import os, subprocess, sys
port = sys.argv[1]
logfile = sys.argv[2]
extra = [a for a in sys.argv[3:] if a]
out = open(logfile, 'ab', buffering=0)
p = subprocess.Popen(
    [sys.executable, 'web/server.py'] + (extra or ['--port', port]),
    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True, cwd=os.getcwd())
print(p.pid)
PY
)
    echo "$NEWPID" > "$PIDFILE"
    # 起得来才算成功
    for _ in $(seq 1 40); do
      if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/"; then
        echo "✅ 已在后台启动 :$PORT (pid $NEWPID)"
        echo "   http://127.0.0.1:$PORT/"
        echo "   日志 $LOGFILE"
        exit 0
      fi
      sleep 0.25
    done
    echo "❌ 启动失败，看 $LOGFILE：" >&2
    tail -20 "$LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
    ;;

  *)
    exec .venv/bin/python web/server.py "${ARGS[@]:---port $PORT}"
    ;;
esac
