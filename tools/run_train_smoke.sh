#!/usr/bin/env bash
# Xin TPU v6e-1 trên Colab -> chạy run_train_smoke.sh (LoRA Gemma 4 E4B) -> tải kết quả -> trả runtime.
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8"
say() { echo "[$(date '+%F %T')] $*"; }
if [ ! -f .colab_runtime.json ]; then
  say "SMOKE_WAIT xin ${ACC:-V6E1} mỗi ${INTERVAL:-10} phút, tối đa ${MAX_HOURS:-10} giờ"
  $PYRUN tools/watch_gpu_api.py --gpu "${ACC:-V6E1}" --interval "${INTERVAL:-10}" --max-hours "${MAX_HOURS:-10}" 2>&1 \
    | grep -E "STATUS=(ACQUIRED|TIMEOUT|FATAL)|Bỏ qua|Không có"
  [ -f .colab_runtime.json ] || { say "SMOKE_STOP không xin được TPU"; exit 1; }
  sleep 20
fi
for i in 1 2 3 4; do
  out=$($PYRUN tools/colab_job.py start --runtime text2asset --script run_train_smoke.sh 2>&1 | tail -1)
  echo "$out" | grep -q JOB_STARTED && break
  say "start lỗi (lần $i): ${out:0:200}"; sleep 45
done
echo "$out" | grep -q JOB_STARTED || { say "SMOKE_STOP không khởi động được job"; $PYRUN tools/watch_gpu_api.py --release --name text2asset; exit 1; }
say "SMOKE_JOB_STARTED"
$PYRUN tools/colab_job.py monitor --runtime text2asset --interval 30 --sync-every 0
rc=$?
say "SMOKE_MONITOR_EXIT rc=$rc"
$PYRUN tools/colab_job.py fetch --runtime text2asset --dest results/train_smoke --rels train_smoke,run.log 2>&1 | tail -1
$PYRUN tools/watch_gpu_api.py --release --name text2asset 2>&1 | tail -1
say "SMOKE_END rc=$rc"
