#!/usr/bin/env bash
# Mốc 0 bổ sung: Inkling-2-Mini (bản .strata SQ-20, không mất mát) qua engine Strata, thinking BẬT, trên L4.
# Strata tự offload: KV cache cấp phát trước trên GPU, expert nào không vừa VRAM thì để ở RAM (MoE 26.4B, 3.4B active).
# Biến: PAR (số request song song, mặc định 4), CTX (ngữ cảnh mỗi slot, 40960), EFFORT (high), SETTINGS (text,image)
# Marker: STAGE / WARN / PHASE0_DONE / PHASE0_FAILED
set -uo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${WORK:-/content/t2a}
TAG=inkling2mini_sq20
PAR=${PAR:-4}; CTX=${CTX:-40960}; EFFORT=${EFFORT:-high}; SETTINGS=${SETTINGS:-text,image}
BL=/content/blender/blender
PY=python3
PORT=8080
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
stage() { echo "[$(date '+%F %T')] STAGE $*"; }
warn()  { echo "[$(date '+%F %T')] WARN $*"; }
fail()  { echo "[$(date '+%F %T')] PHASE0_FAILED $*"; strata launch --stop >/dev/null 2>&1; exit 1; }

stage "0 env nproc=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G par=$PAR ctx=$CTX effort=$EFFORT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "không có GPU"
mkdir -p "$W/results" "$W/gen" "$W/gen_aside"
for f in "$W"/gen/*.jsonl; do [ -e "$f" ] && [ "$(basename "$f" .jsonl)" != "$TAG" ] && mv "$f" "$W/gen_aside/"; done

stage "1 setup blender (nền)"
(
  if [ ! -x "$BL" ]; then
    apt-get -qq update >/dev/null 2>&1
    apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libegl1 >/dev/null 2>&1
    V=$(curl -s https://download.blender.org/release/Blender5.0/ | grep -oE 'blender-5\.0\.[0-9]+-linux-x64\.tar\.xz' | sort -uV | tail -1)
    curl -sL "https://download.blender.org/release/Blender5.0/$V" -o /tmp/blender.tar.xz \
      && mkdir -p /content/blender && tar -xf /tmp/blender.tar.xz -C /content/blender --strip-components=1 && rm -f /tmp/blender.tar.xz
  fi
  # scipy cho script GT (vd CauliflowerCoral): cài bằng pip của Blender, không được thì pip hệ thống --target.
  BPY=$(ls /content/blender/*/python/bin/python3.* 2>/dev/null | grep -v config | head -1)
  if [ -n "$BPY" ] && ! "$BPY" -c "import scipy" 2>/dev/null; then
    { "$BPY" -m ensurepip && "$BPY" -m pip install -q scipy; } > /content/blender_scipy.log 2>&1 || {
      SP=$("$BPY" -c "import site; print(site.getsitepackages()[0])")
      PV=$("$BPY" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
      $PY -m pip install -q --no-deps --only-binary=:all: --python-version "$PV" --target "$SP" "scipy<1.16" >> /content/blender_scipy.log 2>&1
    }
  fi
  echo "BLENDER_READY"
) > /content/blender_setup.log 2>&1 &
$PY -m pip -q install huggingface_hub pillow scipy requests >/dev/null 2>&1 || warn "pip deps"

stage "2 tải benchmark (50 task + ảnh)"
[ -s "$W/tasks.jsonl" ] && [ -d "$W/bench/3DCodeBench_ModelLogs" ] || $PY "$CODE/data_prep.py" --work "$W" --bench-only || fail "data_prep"

stage "3 cài Strata + tải Inkling-2-Mini SQ-20 (~20 GB)"
if ! command -v strata >/dev/null; then
  curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o /tmp/strata_install.sh \
    && sh /tmp/strata_install.sh || fail "cài Strata"
fi
strata launch --plan --family inkling-2-mini --model SQ-20 --context "$CTX" --parallel "$PAR" --vision on 2>&1 | tail -15
strata launch --family inkling-2-mini --model SQ-20 --context "$CTX" --parallel "$PAR" --vision on \
  --port "$PORT" --no-api-key --yes --no-tui > /content/strata_launch.log 2>&1 \
  || { tail -30 /content/strata_launch.log; fail "strata launch"; }
tail -8 /content/strata_launch.log
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 10; done
MID=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | $PY -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])") \
  || fail "server Strata không trả lời"
strata launch --status 2>&1 | tail -12
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "model id: $MID"

stage "5 gen $TAG (thinking=$EFFORT)"
timeout 6h $PY "$CODE/zeroshot_gen_llamacpp.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$MID" --tag "$TAG" \
  --work "$W" --settings "$SETTINGS" --thinking on --parallel "$PAR" --no-template-kwargs \
  --extra-body "{\"reasoning_effort\": \"$EFFORT\"}" --max-tokens-on 32768 2>&1
[ -s "$W/gen/$TAG.jsonl" ] || fail "MODEL_FAILED $TAG"
strata launch --stop >/dev/null 2>&1

stage "6 chờ Blender"
for i in $(seq 1 60); do grep -q BLENDER_READY /content/blender_setup.log 2>/dev/null && break; sleep 10; done
"$BL" -b --factory-startup --version 2>/dev/null | head -1 || fail "Blender không chạy"
BPY=$(ls /content/blender/*/python/bin/python3.* 2>/dev/null | grep -v config | head -1)
"$BPY" -c "import scipy; print('blender scipy', scipy.__version__)" 2>&1 || { tail -5 /content/blender_scipy.log; warn "Blender thiếu scipy"; }

stage "7 render GT + code sinh ra"
$PY "$CODE/score.py" --work "$W" --gt-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gt.jsonl" --render --timeout 300 --threads 2 --label gt
$PY "$CODE/score.py" --work "$W" --gen-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gen.jsonl" --render --timeout 240 --threads 2 --label gen

stage "8 chấm điểm"
$PY "$CODE/score.py" --work "$W" --score || fail "score"
echo "[$(date '+%F %T')] PHASE0_DONE"
