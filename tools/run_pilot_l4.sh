#!/usr/bin/env bash
# Pilot agent 1 đề trên runtime L4 đang có: dừng job zero-shot (giữ mẫu), chạy run_agent_pilot.sh, tải kết quả, trả runtime.
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8"
say() { echo "[$(date '+%F %T')] $*"; }
rexec() { timeout 200 $PYRUN tools/colab_job.py --runtime text2asset exec "$1" 2>&1 | tail -3; }

if [ ! -f .colab_runtime.json ]; then
  say "PILOT_WAIT xin ${GPUS:-L4} mỗi ${INTERVAL:-10} phút, tối đa ${MAX_HOURS:-10} giờ"
  $PYRUN tools/watch_gpu_api.py --gpu "${GPUS:-L4}" --interval "${INTERVAL:-10}" --max-hours "${MAX_HOURS:-10}" 2>&1 | grep -E "STATUS=(ACQUIRED|TIMEOUT|FATAL)"
  [ -f .colab_runtime.json ] || { say "PILOT_STOP không xin được GPU"; exit 1; }
  sleep 20
fi
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'colab_job.py monitor|run_inkling.sh|restart_inkling.sh' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
say "PILOT dừng job zero-shot, lưu mẫu"
rexec "import subprocess; print(subprocess.run(\"pkill -f 'zeroshot_gen_llamacp[p]'; pkill -f 'run_inkling_strat[a]'; sleep 1; wc -l < /content/t2a/gen/inkling2mini_sq20.jsonl\", shell=True, capture_output=True, text=True).stdout)"
timeout 300 $PYRUN tools/colab_job.py fetch --runtime text2asset --dest results/phase0_sync 2>&1 | tail -1
rexec "import os, time; os.path.exists('/content/t2a/run.log') and os.replace('/content/t2a/run.log', f'/content/t2a/run_{int(time.time())}.log'); print('rotated')"
out=$($PYRUN tools/colab_job.py start --runtime text2asset --script run_agent_pilot.sh ${TASK:+--env "TASK=$TASK"} 2>&1 | tail -1)
echo "$out" | grep -q JOB_STARTED || { say "PILOT_STOP không khởi động được: ${out:0:200}"; exit 1; }
say "PILOT_JOB_STARTED"
$PYRUN tools/colab_job.py monitor --runtime text2asset --interval 30 --sync-every 0
rc=$?
say "PILOT_MONITOR_EXIT rc=$rc"
$PYRUN tools/colab_job.py fetch --runtime text2asset --dest results/agent_pilot_l4 --rels pilot,run.log 2>&1 | tail -1
$PYRUN tools/watch_gpu_api.py --release --name text2asset 2>&1 | tail -1
say "PILOT_END rc=$rc"
