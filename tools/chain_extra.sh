#!/usr/bin/env bash
# Chuỗi bổ sung trên runtime đang có: chờ job 27B NariQ xong -> chạy Gemma 4 12B + 31B (w4a16)
# -> tải kết quả về results/phase0 -> trả runtime. Sự kiện in ra stdout.
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8 -W ignore -u"
ts() { date '+%F %T'; }
release() {
  $PYRUN tools/colab_job.py fetch --runtime text2asset --dest results/phase0 2>&1 | tail -1
  $PYRUN tools/watch_gpu_api.py --release --name text2asset 2>&1 | tail -1
}

echo "[$(ts)] CHAIN chờ job Qwen3.8-27B NariQ"
$PYRUN tools/colab_job.py monitor --runtime text2asset --interval 90 --from-end
rc=$?
[ $rc = 4 ] && { echo "[$(ts)] CHAIN_STOP runtime mất (kết quả dở ở results/phase0_sync)"; exit 4; }
[ $rc != 0 ] && echo "[$(ts)] CHAIN 27B kết thúc rc=$rc — vẫn chạy tiếp Gemma"

echo "[$(ts)] CHAIN khởi chạy Gemma 4 12B + 31B-w4"
$PYRUN tools/colab_job.py start --runtime text2asset --env MODELS=gemma4_12b,gemma4_31b_w4 2>&1 | tail -1
$PYRUN tools/colab_job.py monitor --runtime text2asset --interval 90 --from-end
rc=$?
[ $rc = 4 ] && { echo "[$(ts)] CHAIN_STOP runtime mất"; exit 4; }
release
echo "[$(ts)] CHAIN_DONE rc=$rc"
