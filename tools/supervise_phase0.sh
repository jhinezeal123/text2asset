#!/usr/bin/env bash
# Supervisor Mốc 0: xin GPU -> chạy job -> theo dõi (sao lưu kết quả mỗi 10 phút).
# Runtime bị thu hồi -> xin lại & chạy tiếp từ kết quả đã sao lưu (tối đa MAX_ROUNDS lần).
# Xong/lỗi -> tải kết quả về results/phase0 và TRẢ runtime để không tốn compute units.
# Mọi dòng sự kiện in ra stdout (chạy kèm: >> supervise.log).
cd "$(dirname "$0")/.." || exit 1
PYRUN="python -I -X utf8 -W ignore -u"
MAX_ROUNDS=${MAX_ROUNDS:-5}
ts() { date '+%F %T'; }

finish() {  # $1 = nhãn
  $PYRUN tools/colab_job.py fetch --runtime text2asset --dest results/phase0 2>&1 | tail -1
  $PYRUN tools/watch_gpu_api.py --release --name text2asset 2>&1 | tail -1
  echo "[$(ts)] $1"
}

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "[$(ts)] SUP round $round: chờ GPU"
  if ! bash tools/auto_phase0.sh; then echo "[$(ts)] SUP_STOP không xin/khởi chạy được"; exit 1; fi
  $PYRUN tools/colab_job.py monitor --runtime text2asset --interval 90 --from-end
  rc=$?
  case $rc in
    0) finish "SUP_DONE"; exit 0 ;;
    1) finish "SUP_FAILED (xem results/phase0/run.log)"; exit 1 ;;
    4) echo "[$(ts)] SUP_RUNTIME_LOST — đã có bản sao lưu ở results/phase0_sync, xin máy lại" ;;
    *) echo "[$(ts)] SUP monitor thoát rc=$rc"; finish "SUP_STOP"; exit "$rc" ;;
  esac
done
echo "[$(ts)] SUP_STOP quá $MAX_ROUNDS lần mất runtime"
exit 1
