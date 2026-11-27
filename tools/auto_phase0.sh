#!/usr/bin/env bash
# Xin GPU (mặc định L4; A100 chỉ dùng khi train: GPUS=A100:hm,A100) -> tải code lên và chạy Mốc 0 ngay.
# Thoát khi: job đã khởi chạy (AUTO_JOB_STARTED) | canh thất bại/hết giờ (AUTO_STOP).
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8 -W ignore"
$PYRUN tools/watch_gpu_api.py --gpu "${GPUS:-L4}" --interval "${INTERVAL:-10}" --max-hours "${MAX_HOURS:-12}" \
  | grep --line-buffered "STATUS=" | grep --line-buffered -v "STATUS=RETRY"
st=${PIPESTATUS[0]}
if [ "$st" != 0 ]; then echo "AUTO_STOP watcher exit=$st"; exit "$st"; fi
for i in 1 2 3; do
  if $PYRUN tools/colab_job.py start --runtime text2asset; then echo "AUTO_JOB_STARTED"; exit 0; fi
  echo "AUTO_START_RETRY $i"; sleep 30
done
echo "AUTO_STOP không khởi chạy được job"; exit 5
