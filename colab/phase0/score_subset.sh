#!/usr/bin/env bash
# Chấm điểm lại một phần kết quả sinh code trên runtime CHỈ CPU (không cần GPU):
# chỉ giữ gen/<tag>.jsonl trong TAGS, chỉ tải + render GT cho các task có trong đó, rồi render + chấm điểm.
# Marker: STAGE / WARN / PHASE0_DONE / PHASE0_FAILED
set -uo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${WORK:-/content/t2a}
TAGS=${TAGS:-qwen38_27b_nariq3}
BL=/content/blender/blender
PY=python3
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
stage() { echo "[$(date '+%F %T')] STAGE $*"; }
warn()  { echo "[$(date '+%F %T')] WARN $*"; }
fail()  { echo "[$(date '+%F %T')] PHASE0_FAILED $*"; exit 1; }

stage "0 env nproc=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G tags=$TAGS"
mkdir -p "$W/results" "$W/gen_aside"
for f in "$W"/gen/*.jsonl; do
  t=$(basename "$f" .jsonl)
  case ",$TAGS," in *",$t,"*) ;; *) mv "$f" "$W/gen_aside/" ;; esac
done
ls "$W"/gen/*.jsonl >/dev/null 2>&1 || fail "không có gen/*.jsonl cho TAGS=$TAGS"
IDS=$($PY -c "import json,glob; print(','.join(sorted({json.loads(l)['task'] for f in glob.glob('$W/gen/*.jsonl') for l in open(f)})))")
echo "task: $(echo "$IDS" | tr ',' '\n' | wc -l)"

stage "1 setup blender"
if [ ! -x "$BL" ]; then
  apt-get -qq update >/dev/null 2>&1
  apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libegl1 >/dev/null 2>&1 || warn "apt deps"
  V=$(curl -s https://download.blender.org/release/Blender5.0/ | grep -oE 'blender-5\.0\.[0-9]+-linux-x64\.tar\.xz' | sort -uV | tail -1)
  [ -n "$V" ] || fail "không tìm thấy bản Blender 5.0"
  curl -sL "https://download.blender.org/release/Blender5.0/$V" -o /tmp/blender.tar.xz || fail "tải Blender"
  mkdir -p /content/blender && tar -xf /tmp/blender.tar.xz -C /content/blender --strip-components=1 || fail "giải nén Blender"
  rm -f /tmp/blender.tar.xz
fi
"$BL" -b --factory-startup --version 2>/dev/null | head -1 || fail "Blender không chạy"
$PY -m pip -q install huggingface_hub pillow scipy >/dev/null 2>&1 || warn "pip deps"
$PY -c "import torch, transformers" 2>/dev/null || $PY -m pip -q install torch transformers >/dev/null 2>&1 || warn "torch/transformers"

stage "2 tải benchmark (chỉ các task cần)"
$PY "$CODE/data_prep.py" --work "$W" --bench-only --only "$IDS" || fail "data_prep"

stage "3 render ground truth"
$PY "$CODE/score.py" --work "$W" --gt-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gt.jsonl" --render --timeout 300 --threads 2 --label gt

stage "7 render code sinh ra"
$PY "$CODE/score.py" --work "$W" --gen-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$W/manifest_gen.jsonl" --render --timeout 240 --threads 2 --label gen

stage "8 chấm điểm"
$PY "$CODE/score.py" --work "$W" --score || fail "score"
echo "[$(date '+%F %T')] PHASE0_DONE"
