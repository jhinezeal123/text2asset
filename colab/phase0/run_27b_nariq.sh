#!/usr/bin/env bash
# Bổ sung Qwen3.8-27B (NariQ 3-bit GGUF qua llama.cpp fork) vào kết quả Mốc 0 đang có trên VM.
# Cần: /content/nari_setup.sh đã chạy (bin + model). Marker: STAGE / WARN / PHASE0_DONE / PHASE0_FAILED
set -uo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${WORK:-/content/t2a}
BL=/content/blender/blender
PY=python3
TAG=qwen38_27b_nariq3
stage() { echo "[$(date '+%F %T')] STAGE $*"; }
warn()  { echo "[$(date '+%F %T')] WARN $*"; }
fail()  { echo "[$(date '+%F %T')] PHASE0_FAILED $*"; exit 1; }

stage "27b-1 chờ bin + model NariQ"
if ! grep -q MODEL_READY /content/nari_setup.log 2>/dev/null && ! pgrep -f nari_setup.sh >/dev/null; then
  bash "$CODE/nari_setup.sh" > /content/nari_setup.log 2>&1 &   # VM mới: tự tải
fi
for i in $(seq 1 180); do grep -q MODEL_READY /content/nari_setup.log 2>/dev/null && break; sleep 10; done
grep -q MODEL_READY /content/nari_setup.log || { tail -5 /content/nari_setup.log; fail "tải NariQ chưa xong/lỗi"; }
SERVER=$(find /content/nari/bin -name llama-server -type f | head -1)
[ -x "$SERVER" ] || fail "không thấy llama-server"
# cudart/cublas nằm ở thư mục riêng -> chép cạnh llama-server, nếu không libggml-cuda.so không nạp được và chạy CPU.
cp -n /content/nari/bin/cudart-*/lib* "$(dirname "$SERVER")/" 2>/dev/null
LD_LIBRARY_PATH="$(dirname "$SERVER"):${LD_LIBRARY_PATH:-}" "$SERVER" --list-devices 2>&1 | tail -3
LD_LIBRARY_PATH="$(dirname "$SERVER"):${LD_LIBRARY_PATH:-}" "$SERVER" --list-devices 2>&1 | grep -q CUDA0 \
  || fail "llama-server không thấy GPU CUDA"
MODEL=/content/nari/model/Qwen3.8-27B-NariQ-3.0bpw.gguf
MMPROJ=/content/nari/model/Qwen3.8-27B-NariQ-mmproj-BF16.gguf

stage "5 gen $TAG"
$PY -m pip -q install requests pillow >/dev/null 2>&1
timeout 5h $PY "$CODE/zeroshot_gen_llamacpp.py" --server "$SERVER" --model "$MODEL" --mmproj "$MMPROJ" \
  --tag "$TAG" --work "$W" --thinking off,on 2>&1
[ -s "$W/gen/$TAG.jsonl" ] || { tail -30 "$W/llama_server_$TAG.log"; fail "MODEL_FAILED $TAG"; }

stage "7 render code sinh ra (chỉ phần mới)"
$PY "$CODE/score.py" --work "$W" --gen-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gen.jsonl" --render --timeout 240 --threads 2 --label gen

stage "8 chấm điểm"
$PY "$CODE/score.py" --work "$W" --score || fail "score"
echo "[$(date '+%F %T')] PHASE0_DONE"
