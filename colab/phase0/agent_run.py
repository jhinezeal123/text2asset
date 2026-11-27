"""Chạy model trong mini-swe-agent (chế độ tool-calling: model gọi tool `bash`) để viết script Blender cho từng đề.

python agent_run.py --base-url http://127.0.0.1:8090/v1 --model inkling-2-mini-sq-20 --tag inkling_agent \
    --work /content/t2a_pilot --blender /content/blender/blender --tasks Auger_seed0 --step-limit 30
Mỗi đề có thư mục riêng <work>/agent/<tag>/<task>/ (model.py + trajectory.json). Kết quả cuối ghi vào
gen/<tag>.jsonl cùng định dạng zeroshot_gen.py (code = model.py cuối cùng) để score.py chấm được.
Agent chỉ thấy lệnh `check-model` (bộ chấm thật được chép ra chỗ khác, không nằm trong thư mục làm việc).
"""

import argparse
import json
import os
import shutil
import stat
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
GRADER_DIR = "/opt/t2a_grader"
CHECK_BIN = "/usr/local/bin/check-model"

INSTANCE = """{{task}}

## Environment
- Linux shell; every bash call runs in a fresh subshell inside your working directory. Blender 5.0 is at `{blender}`.
- Deliverable: a Blender 5.0 Python script saved as `model.py` in the working directory. It is run headless
  (`blender -b --factory-startup --python model.py`) and must build the object as mesh geometry.
  Do not add cameras, lights, world or render settings: the grader renders the object itself.

## Workflow (keep it short)
1. Write a first complete `model.py` early (create files with a heredoc in ONE bash call, real newlines):
   cat <<'EOF' > model.py
   import bpy
   ...
   EOF
2. Run `check-model` — it executes `model.py` exactly like the grader and prints `OK tris=N`, or the error with traceback.
3. Fix errors and re-check. Only look up the API when an error tells you something is wrong, e.g.
   `{blender} -b --factory-startup --python-expr "import bmesh; help(bmesh.ops.create_cone)" 2>/dev/null | head -30`
4. Once `check-model` prints OK and the shape matches the description, finish with
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (alone, in its own bash call).
"""


def install_grader(blender):
    """Chép blender_exec.py ra ngoài thư mục code và tạo lệnh check-model gọn cho agent."""
    os.makedirs(GRADER_DIR, exist_ok=True)
    shutil.copy(os.path.join(HERE, "blender_exec.py"), os.path.join(GRADER_DIR, "blender_exec.py"))
    script = f"""#!/bin/sh
F="${{1:-model.py}}"; O=$(mktemp -d)
[ -f "$F" ] || {{ echo "no $F in $(pwd)"; exit 1; }}
timeout 240 {blender} -b --factory-startup --python {GRADER_DIR}/blender_exec.py -- "$F" "$O" >"$O/log.txt" 2>&1
python3 - "$O" <<'PY'
import json, os, sys
o = sys.argv[1]; p = os.path.join(o, "result.json")
if not os.path.exists(p):
    print("Blender crashed or timed out:"); print(open(os.path.join(o, "log.txt"), errors="replace").read()[-1500:]); sys.exit(1)
r = json.load(open(p))
if r.get("ok"):
    print(f"OK tris={{r['tris']}} mesh_objects={{r.get('n_mesh_objects')}}")
else:
    print((r.get("traceback") or r.get("error") or "the script ran but built no mesh geometry (0 triangles)")[-2500:]); sys.exit(1)
PY
"""
    Path(CHECK_BIN).write_text(script)
    os.chmod(CHECK_BIN, os.stat(CHECK_BIN).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--blender", default="/content/blender/blender")
    p.add_argument("--tasks", default="")
    p.add_argument("--step-limit", type=int, default=30)
    p.add_argument("--wall-limit", type=int, default=2700, help="giây tối đa mỗi đề")
    p.add_argument("--max-tokens", type=int, default=12288, help="giới hạn output mỗi lượt (thinking + trả lời)")
    p.add_argument("--think-budget", type=int, default=8192, help="giới hạn thinking mỗi lượt")
    p.add_argument("--effort", default="high")
    a = p.parse_args()

    import yaml
    import minisweagent
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model

    base_cfg = yaml.safe_load((Path(minisweagent.__file__).parent / "config" / "mini.yaml").read_text())
    install_grader(a.blender)
    tasks = [json.loads(l) for l in open(os.path.join(a.work, "tasks.jsonl"), encoding="utf-8")]
    if a.tasks:
        keep = set(a.tasks.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    sysp = open(os.path.join(a.work, "bench", "text_to_3d_system_prompt.txt"), encoding="utf-8").read()
    out = os.path.join(a.work, "gen", f"{a.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    for t in tasks:
        d = Path(a.work) / "agent" / a.tag / t["id"]
        d.mkdir(parents=True, exist_ok=True)
        task_text = sysp.strip() + "\n\n# Object to build\n" + t["text_prompt"]  # biến Jinja {{task}}
        model = get_model(input_model_name=f"openai/{a.model}", config={
            "model_kwargs": {"api_base": a.base_url, "api_key": "none", "drop_params": True, "temperature": 0.7,
                             "top_p": 0.95, "max_tokens": a.max_tokens,
                             "extra_body": {"reasoning_effort": a.effort, "reasoning_budget_tokens": a.think_budget}},
            "cost_tracking": "ignore_errors",
            "observation_template": base_cfg["model"]["observation_template"],
            "format_error_template": base_cfg["model"]["format_error_template"],
        })
        env = LocalEnvironment(cwd=str(d), env=base_cfg["environment"]["env"], timeout=600)
        agent = DefaultAgent(model, env, system_template=base_cfg["agent"]["system_template"],
                             instance_template=INSTANCE.replace("{blender}", a.blender),
                             step_limit=a.step_limit, cost_limit=0, wall_time_limit_seconds=a.wall_limit,
                             output_path=d / "trajectory.json")
        t0 = time.time()
        try:
            res = agent.run(task_text)
        except Exception as e:  # agent hỏng giữa chừng: vẫn chấm model.py nếu có
            res = {"exit_status": f"crash {type(e).__name__}: {str(e)[:200]}"}
        dt = time.time() - t0
        code = (d / "model.py").read_text(encoding="utf-8", errors="replace") if (d / "model.py").exists() else ""
        n_check = sum(1 for m in agent.messages if m.get("role") == "assistant" and "check-model" in json.dumps(m.get("extra", {})))
        rec = {"tag": a.tag, "model": a.model, "task": t["id"], "setting": "text", "thinking": "on", "raw": "",
               "code": code, "out_tokens": 0, "finish_reason": res.get("exit_status"), "steps": agent.n_calls,
               "checks": n_check, "seconds": round(dt, 1), "exit_status": res.get("exit_status")}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{a.tag}] {t['id']}: {res.get('exit_status')} sau {agent.n_calls} bước ({n_check} lần check), {dt:.0f}s, "
              f"model.py {len(code)} ký tự", flush=True)


if __name__ == "__main__":
    main()
