#!/usr/bin/env bash
# Pilot 1 đề: Inkling-2-Mini SQ-20 (Strata, 1 request, ngữ cảnh 64K) chạy trong mini-swe-agent, rồi render + chấm đúng đề đó.
# Biến: TASK (Auger_seed0), STEPS (30), CTX (196608 = 192K), KV (fp8). Marker: STAGE / WARN / PHASE0_DONE / PHASE0_FAILED
set -uo pipefail
CODE="$(cd "$(dirname "$0")" && pwd)"
TASK=${TASK:-Auger_seed0}; STEPS=${STEPS:-30}; CTX=${CTX:-196608}; KV=${KV:-fp8}
W=/content/t2a; P=/content/t2a_pilot; BL=/content/blender/blender; PORT=8090; PY=python3; TAG=inkling_agent
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
stage() { echo "[$(date '+%F %T')] STAGE $*"; }
warn()  { echo "[$(date '+%F %T')] WARN $*"; }
fail()  { echo "[$(date '+%F %T')] PHASE0_FAILED $*"; strata launch --stop >/dev/null 2>&1; exit 1; }

stage "setup (máy mới thì tự cài Blender, dữ liệu đề, Strata)"
if [ ! -x "$BL" ]; then
  apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libegl1 >/dev/null 2>&1
  V=$(curl -s https://download.blender.org/release/Blender5.0/ | grep -oE 'blender-5\.0\.[0-9]+-linux-x64\.tar\.xz' | sort -uV | tail -1)
  curl -sL "https://download.blender.org/release/Blender5.0/$V" -o /tmp/b.tar.xz && mkdir -p /content/blender \
    && tar -xf /tmp/b.tar.xz -C /content/blender --strip-components=1 && rm -f /tmp/b.tar.xz || fail "cài Blender"
fi
"$BL" -b --factory-startup --version 2>/dev/null | head -1
$PY -m pip -q install huggingface_hub pillow scipy requests >/dev/null 2>&1
grep -q "\"id\": \"$TASK\"" "$W/tasks.jsonl" 2>/dev/null || $PY "$CODE/data_prep.py" --work "$W" --bench-only --only "$TASK" || fail "data_prep"
command -v strata >/dev/null || { curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o /tmp/si.sh && sh /tmp/si.sh | tail -1; } || fail "cài Strata"

stage "0 Strata: 1 request, ctx $CTX, KV $KV, không đọc ảnh"
pkill -f 'zeroshot_gen_llamacp[p]'; pkill -f 'run_inkling_strat[a]'; sleep 2
strata launch --family inkling-2-mini --model SQ-20 --context "$CTX" --parallel 1 --vision off \
  --port "$PORT" --no-api-key --yes --no-tui --restart --set "kv=$KV" > /content/strata_pilot.log 2>&1 || { tail -20 /content/strata_pilot.log; fail "strata launch"; }
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 5; done
MID=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | $PY -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])") || fail "server không lên"
grep -hE "engine started|Loaded|Host KV|memory:" "$HOME"/.local/state/strata/launch/*.engine.log | tail -4
curl -sf "http://127.0.0.1:$PORT/metrics" | $PY -c "import json,sys; e=json.load(sys.stdin)['engine']; print('kv_dtype', e.get('kv_dtype'), 'max_context', e.get('max_context'))"
[ -x "$BL" ] || fail "không thấy Blender"

stage "1 thư mục pilot riêng (chỉ đề $TASK)"
rm -rf "$P"; mkdir -p "$P/gen" "$P/results"
ln -s "$W/bench" "$P/bench"
grep "\"id\": \"$TASK\"" "$W/tasks.jsonl" > "$P/tasks.jsonl" || fail "không thấy đề $TASK"
$PY -m pip -q install mini-swe-agent pyyaml >/dev/null 2>&1 || fail "cài mini-swe-agent"
$PY -c "import minisweagent; print('mini-swe-agent', minisweagent.__version__)"

stage "2 agent (tool-calling): $TASK (tối đa $STEPS bước, 45 phút)"
T0=$(date +%s)
$PY "$CODE/agent_run.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$MID" --tag "$TAG" --work "$P" \
  --blender "$BL" --tasks "$TASK" --step-limit "$STEPS" --wall-limit 2700 2>&1 | grep -vE "^\s*$" | tail -40
echo "agent_wall_s=$(( $(date +%s) - T0 ))"
curl -sf "http://127.0.0.1:$PORT/metrics" > "$P/results/strata_metrics.json"
$PY - "$P" <<'EOF'
import json, sys
p = sys.argv[1]
m = json.load(open(f"{p}/results/strata_metrics.json"))
reqs = m.get("requests", [])
out = sum(r.get("output_tokens") or 0 for r in reqs); dur = sum(r.get("duration_s") or 0 for r in reqs)
s = {"requests": len(reqs), "output_tokens": out, "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in reqs),
     "gen_seconds": round(dur, 1), "decode_tok_s": round(out / max(dur, 1), 1)}
json.dump(s, open(f"{p}/results/pilot_stats.json", "w"), indent=2)
print("PILOT_STATS", json.dumps(s))
EOF
strata launch --stop >/dev/null 2>&1

stage "3 render + chấm điểm"
$PY "$CODE/score.py" --work "$P" --gt-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$P/manifest_gt.jsonl" --render --timeout 300 --threads 4 --label gt
$PY "$CODE/score.py" --work "$P" --gen-manifest
$PY "$CODE/exec_runner.py" --blender "$BL" --manifest "$P/manifest_gen.jsonl" --render --timeout 240 --threads 4 --label gen
$PY "$CODE/score.py" --work "$P" --score || warn "score"
cp -r "$P/agent" "$P/results/agent" 2>/dev/null
rm -rf "$W/pilot"; mkdir -p "$W/pilot"; cp -r "$P/results" "$P/render" "$P/gen" "$W/pilot/"; find "$W/pilot" -name "*.npy" -delete
echo "[$(date '+%F %T')] PHASE0_DONE"
