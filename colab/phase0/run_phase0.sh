#!/usr/bin/env bash
# Mốc 0 — chạy toàn bộ trên runtime Colab. Mọi dòng quan trọng có marker:
#   STAGE ... | WARN ... | PHASE0_DONE | PHASE0_FAILED ...
# Chạy lại được: các bước bỏ qua phần đã làm xong.
# Biến môi trường: SMOKE=1 (chạy thử cực nhỏ), N_TASKS, N_EXEC, N_TOK, WORK.
set -uo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${WORK:-/content/t2a}
mkdir -p "$W/results"
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then N_TASKS=3; N_EXEC=4; N_TOK=200
else N_TASKS=${N_TASKS:-50}; N_EXEC=${N_EXEC:-300}; N_TOK=${N_TOK:-2000}; fi
BL=/content/blender/blender
VENV=/content/vllm-env
PY=python3
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

stage() { echo "[$(date '+%F %T')] STAGE $*"; }
warn()  { echo "[$(date '+%F %T')] WARN $*"; }
fail()  { echo "[$(date '+%F %T')] PHASE0_FAILED $*"; exit 1; }

stage "0 env"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then HAS_GPU=1; else HAS_GPU=0; fi
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_MB=${GPU_MB:-0}
echo "nproc=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G gpu=$HAS_GPU gpu_mb=$GPU_MB smoke=$SMOKE n_tasks=$N_TASKS n_exec=$N_EXEC"

stage "1 setup blender"
if [ ! -x "$BL" ]; then
  apt-get -qq update >/dev/null 2>&1
  apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libegl1 >/dev/null 2>&1 || warn "apt deps"
  V=$(curl -s https://download.blender.org/release/Blender5.0/ | grep -oE 'blender-5\.0\.[0-9]+-linux-x64\.tar\.xz' | sort -uV | tail -1)
  [ -n "$V" ] || fail "không tìm thấy bản Blender 5.0"
  echo "tải $V"
  curl -sL "https://download.blender.org/release/Blender5.0/$V" -o /tmp/blender.tar.xz || fail "tải Blender"
  mkdir -p /content/blender && tar -xf /tmp/blender.tar.xz -C /content/blender --strip-components=1 || fail "giải nén Blender"
  rm -f /tmp/blender.tar.xz
fi
"$BL" -b --factory-startup --version 2>/dev/null | head -1 || fail "Blender không chạy"
$PY -m pip -q install huggingface_hub pandas pyarrow pillow scipy >/dev/null 2>&1 || warn "pip deps"

if [ "$HAS_GPU" = 1 ]; then
  stage "1b setup vllm"
  if [ ! -x "$VENV/bin/python" ]; then
    $PY -m pip -q install uv >/dev/null 2>&1
    uv venv -q "$VENV" --python 3.12 && uv pip install -q --python "$VENV/bin/python" vllm pillow || warn "cài vllm lỗi"
  fi
  "$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)" || warn "import vllm lỗi"
fi

stage "2 data prep"
if [ ! -f "$W/results/token_stats.json" ]; then
  $PY "$CODE/data_prep.py" --work "$W" --n-tasks "$N_TASKS" --n-exec "$N_EXEC" --n-tok "$N_TOK" || fail "data_prep"
fi

stage "3 render ground truth"
$PY "$CODE/score.py" --work "$W" --gt-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gt.jsonl" --render --timeout 300 --threads 2 --label gt

stage "4 execset (nền, CPU)"
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_execset.jsonl" --timeout 240 --threads 1 \
    --workers $(( $(nproc) > 3 ? $(nproc) / 2 : 1 )) --label execset > "$W/execset.log" 2>&1 &
EXEC_PID=$!

MODELS=${MODELS:-qwen35_4b,qwen35_9b,qwen38_27b,gemma4_26b,gemma4_12b,gemma4_31b_w4}
run_model() {
  local tag=$1; shift
  case ",$MODELS," in *",$tag,"*) ;; *) return 0 ;; esac
  stage "5 gen $tag"
  timeout 3h "$VENV/bin/python" "$CODE/zeroshot_gen.py" --work "$W" --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -vE "it/s\]|^\s*$|Processed prompts|Attempted to load weight" || true
  [ -s "$W/gen/$tag.jsonl" ] || warn "MODEL_FAILED $tag"
}

if [ "$HAS_GPU" = 1 ]; then
  if [ "$GPU_MB" -ge 70000 ]; then GQ=""; else GQ="--quant fp8"; fi
  run_model qwen35_4b  --model Qwen/Qwen3.5-4B --thinking off,on
  run_model qwen35_9b  --model Qwen/Qwen3.5-9B --thinking off,on
  # 27B: bản FP8 (28.8 GB) trên GPU 40 GB chỉ còn ~2 GB KV cache -> không chạy nổi chuỗi dài; cần GPU >= 70 GB (bf16).
  if [ "$GPU_MB" -ge 70000 ]; then
    run_model qwen38_27b --model Qwen/Qwen3.8-27B --thinking off,on
  elif [[ ",$MODELS," == *",qwen38_27b,"* ]]; then
    warn "MODEL_SKIPPED qwen38_27b: GPU ${GPU_MB}MB < 70GB (cần G4/A100-80GB)"
  fi
  # shellcheck disable=SC2086
  run_model gemma4_26b --model google/gemma-4-26B-A4B-it $GQ --thinking off
  # GPU nhỏ (L4 24 GB): 12B bf16 (~24 GB) không vừa -> FP8 (~13 GB); 31B-w4 (~19 GB) gần như không còn KV cache -> bỏ qua.
  if [ "$GPU_MB" -ge 30000 ]; then G12=""; else G12="--quant fp8"; fi
  # shellcheck disable=SC2086
  run_model gemma4_12b --model google/gemma-4-12B-it $G12 --thinking off
  if [ "$GPU_MB" -ge 30000 ]; then
    run_model gemma4_31b_w4 --model google/gemma-4-31B-it-qat-w4a16-ct --thinking off --max-model-len 24576
  elif [[ ",$MODELS," == *",gemma4_31b_w4,"* ]]; then
    warn "MODEL_SKIPPED gemma4_31b_w4: GPU ${GPU_MB}MB < 30GB"
  fi
elif [ "$SMOKE" = 1 ]; then
  warn "không có GPU — SMOKE: dùng code GT làm 'model oracle' để kiểm tra pipeline chấm điểm"
  $PY - "$W" <<'EOF'
import json, os, sys
w = sys.argv[1]; os.makedirs(f"{w}/gen", exist_ok=True)
with open(f"{w}/gen/oracle_gt.jsonl", "w") as f:
    for l in open(f"{w}/tasks.jsonl"):
        t = json.loads(l)
        f.write(json.dumps({"tag": "oracle_gt", "model": "gt", "task": t["id"], "setting": "text", "thinking": "off",
                            "raw": "", "code": open(t["gt_code"]).read(), "out_tokens": 0, "finish_reason": "stop"}) + "\n")
EOF
else
  fail "không có GPU"
fi

stage "6 chờ execset"
wait $EXEC_PID || warn "execset lỗi"
tail -2 "$W/execset.log"

stage "7 render code sinh ra"
$PY "$CODE/score.py" --work "$W" --gen-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gen.jsonl" --render --timeout 240 --threads 2 --label gen

stage "8 chấm điểm"
$PY "$CODE/score.py" --work "$W" --score || fail "score"
echo "[$(date '+%F %T')] PHASE0_DONE"
