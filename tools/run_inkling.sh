#!/usr/bin/env bash
# Canh L4 (mặc định 10 phút/lần, tối đa 10 giờ) -> chạy run_inkling_strata.sh -> theo dõi -> tải kết quả + ảnh -> trả runtime.
# Log: inkling.log (dòng có marker INKLING_* / STAGE / [gen] để Monitor bắt).
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8"
DEST=results/inkling_sq20
say() { echo "[$(date '+%F %T')] $*"; }

say "INKLING_WAIT xin ${GPUS:-L4} mỗi ${INTERVAL:-10} phút, tối đa ${MAX_HOURS:-10} giờ"
$PYRUN tools/watch_gpu_api.py --gpu "${GPUS:-L4}" --interval "${INTERVAL:-10}" --max-hours "${MAX_HOURS:-10}" 2>&1 | grep -E "STATUS=(ACQUIRED|TIMEOUT|FATAL)"
$PYRUN tools/watch_gpu_api.py --list >/dev/null 2>&1
[ -f .colab_runtime.json ] || { say "INKLING_STOP không xin được GPU"; exit 1; }
say "INKLING_ACQUIRED"

for i in 1 2 3 4; do
  out=$($PYRUN tools/colab_job.py start --runtime text2asset --script run_inkling_strata.sh ${STRATA_SET:+--env "STRATA_SET=$STRATA_SET"} 2>&1 | tail -1)
  echo "$out" | grep -q JOB_STARTED && { say "INKLING_JOB_STARTED"; break; }
  say "start lỗi (lần $i): ${out:0:200}"; sleep 60
done
echo "$out" | grep -q JOB_STARTED || { say "INKLING_STOP không khởi động được job"; $PYRUN tools/watch_gpu_api.py --release --name text2asset; exit 1; }

$PYRUN tools/colab_job.py monitor --runtime text2asset --interval 60 --sync-every 900
rc=$?
say "INKLING_MONITOR_EXIT rc=$rc"
$PYRUN tools/colab_job.py fetch --runtime text2asset --dest "$DEST" --with-render 2>&1 | tail -1
$PYRUN tools/watch_gpu_api.py --release --name text2asset 2>&1 | tail -1
say "INKLING_END rc=$rc"
