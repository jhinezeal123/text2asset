#!/usr/bin/env bash
# Chạy thử LoRA Gemma 4 E4B trên Colab: TPU (v6e-1) hoặc GPU (A100), tự nhận biết. Marker: STAGE / WARN / PHASE0_DONE / PHASE0_FAILED
set -uo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
W=/content/t2a_train
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
stage() { echo "[$(date '+%F %T')] STAGE $*"; }
fail()  { echo "[$(date '+%F %T')] PHASE0_FAILED $*"; exit 1; }

stage "0 env"
python3 --version; nproc; free -g | head -2; ls /dev/accel* /dev/vfio 2>/dev/null | head -3
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ACC=gpu; JAXPKG="jax[cuda12]>=0.6.0,!=0.7.2,<0.12"; nvidia-smi -L
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # mặc định JAX chỉ giữ 75% VRAM
else
  ACC=tpu; JAXPKG="jax[tpu]>=0.6.0,!=0.7.2,<0.12"
fi
echo "Máy tăng tốc: $ACC"
python3 -c "import sys; assert sys.version_info >= (3, 12), 'Tunix cần Python >= 3.12'" || fail "Python < 3.12"

stage "1 cài Tunix + JAX ($ACC)"
python3 -m pip install -q "$JAXPKG" "git+https://github.com/google/tunix" qwix \
  transformers huggingface_hub pandas pyarrow pillow 2>&1 | tail -3
python3 -c "import jax, tunix, qwix; print('jax', jax.__version__, jax.devices()); assert jax.devices()[0].platform == '$ACC'"   || fail "import jax/tunix hoặc JAX không thấy $ACC"

stage "2 chạy thử LoRA E4B (ảnh + chữ -> code)"
mkdir -p "$W"
python3 "$CODE/train_smoke_e4b.py" --work "$W" ${SMOKE_ARGS:-} 2>&1 | grep -vE "^\s*$" | grep -v "it/s\]"
mkdir -p /content/t2a/train_smoke && cp "$W/smoke_result.json" /content/t2a/train_smoke/ 2>/dev/null
grep -q '"ok": true' "$W/smoke_result.json" 2>/dev/null || fail "smoke test lỗi (xem smoke_result.json)"
echo "[$(date '+%F %T')] PHASE0_DONE"
