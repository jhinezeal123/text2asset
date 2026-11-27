#!/usr/bin/env bash
# Khởi động lại job Inkling trên runtime đang có với STRATA_SET mới, không mất mẫu đã sinh:
#   WAIT_N=8 STRATA_SET="" bash tools/restart_inkling.sh   (chờ gen đủ WAIT_N dòng rồi mới đổi; STRATA_SET rỗng = mặc định)
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8"
say() { echo "[$(date '+%F %T')] $*"; }
rexec() { timeout 200 $PYRUN tools/colab_job.py --runtime text2asset exec "$1" 2>&1 | tail -5; }
CNT="import subprocess; print('N=' + subprocess.run('wc -l < /content/t2a/gen/inkling2mini_sq20.jsonl', shell=True, capture_output=True, text=True).stdout.strip())"

if [ -n "${WAIT_N:-}" ]; then
  while :; do
    n=$(rexec "$CNT" | grep -oE 'N=[0-9]+' | cut -d= -f2)
    say "INKLING_RESTART chờ: ${n:-?}/${WAIT_N} mẫu"
    [ -n "$n" ] && [ "$n" -ge "$WAIT_N" ] && break
    sleep 60
  done
fi
# 1) dừng trình giám sát cục bộ (nếu không nó sẽ trả runtime khi job cũ báo FAILED)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'colab_job.py monitor|run_inkling.sh' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
# 2) dừng job + Strata trên runtime, lưu gen về máy, đổi tên log cũ
rexec "import subprocess; print(subprocess.run(\"pkill -f 'zeroshot_gen_llamacp[p]'; pkill -f 'run_inkling_strat[a]'; sleep 2; PATH=\$HOME/.local/bin:\$PATH strata launch --stop 2>&1 | tail -1; wc -l < /content/t2a/gen/inkling2mini_sq20.jsonl\", shell=True, capture_output=True, text=True).stdout)"
timeout 300 $PYRUN tools/colab_job.py fetch --runtime text2asset --dest results/phase0_sync 2>&1 | tail -1
rexec "import os, time; os.path.exists('/content/t2a/run.log') and os.replace('/content/t2a/run.log', f'/content/t2a/run_{int(time.time())}.log'); print('rotated')"
say "INKLING_RESTART khởi động lại với STRATA_SET='${STRATA_SET:-}'"
# 3) chạy lại trình giám sát (dùng runtime đang có, tự tải + trả runtime khi xong)
exec bash tools/run_inkling.sh
