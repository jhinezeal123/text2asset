"""Chạy model trong mini-swe-agent (text-based, mỗi lượt 1 lệnh bash) để viết script Blender cho từng đề.

python agent_run.py --base-url http://127.0.0.1:8090/v1 --model inkling-2-mini-sq-20 --tag inkling_agent \
    --work /tmp/t2a --blender /tmp/blender/blender --tasks Auger_seed0 --step-limit 20
Mỗi đề có thư mục riêng <work>/agent/<tag>/<task>/ (model.py + trajectory.json). Kết quả cuối ghi vào
gen/<tag>.jsonl cùng định dạng zeroshot_gen.py (code = model.py cuối cùng) để score.py chấm được.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))

INSTANCE = """{{task}}

## How to work
You are in a Linux shell. Blender 5.0 is at `{blender}` (run it headless with `-b --factory-startup`).
Write the final script to `model.py` in the current directory. Every command runs in a fresh subshell in this directory.

- Test the script exactly as the grader will (background mode, factory settings, no GUI):
  `{check}`
  It prints `OK` with the triangle count, or the error and traceback. The script must build mesh geometry.
- Look up the real Blender 5.0 Python API instead of guessing, for example:
  `{blender} -b --factory-startup --python-expr "import bpy, bmesh; help(bmesh.ops.create_cone)" 2>/dev/null | head -40`
  `{blender} -b --factory-startup --python-expr "import bpy; print(bpy.ops.mesh.primitive_cone_add.__doc__)" 2>/dev/null | tail -5`
- Keep the object recognisable and reasonably detailed; do not add cameras, lights or render settings (the grader renders it).
- When `model.py` runs cleanly and you are satisfied with the shape, finish with: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  (alone, not combined with other commands). You cannot continue after that.

## Response format
Each response: a short THOUGHT, then exactly ONE bash block:

```mswea_bash_command
your_command_here
```

To create or overwrite a file use a heredoc, e.g.
```mswea_bash_command
cat <<'EOF' > model.py
import bpy
...
EOF
```
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--blender", default="/content/blender/blender")
    p.add_argument("--tasks", default="")
    p.add_argument("--setting", default="text", help="hiện chỉ hỗ trợ text (agent nhận mô tả chữ)")
    p.add_argument("--step-limit", type=int, default=20)
    p.add_argument("--wall-limit", type=int, default=3600, help="giây tối đa mỗi đề")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--extra-body", default='{"reasoning_effort": "high"}')
    a = p.parse_args()

    import yaml
    import minisweagent
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model

    base_cfg = yaml.safe_load((Path(minisweagent.__file__).parent / "config" / "mini_textbased.yaml").read_text())
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
        check = (f"{a.blender} -b --factory-startup --python {HERE}/blender_exec.py -- model.py /tmp/chk_{t['id']} >/dev/null 2>&1; "
                 f"python3 -c \"import json; r=json.load(open('/tmp/chk_{t['id']}/result.json')); "
                 f"print('OK tris=%d' % r['tris'] if r['ok'] else (r.get('traceback') or r.get('error') or 'no mesh (0 triangles)')[-2500:])\"")
        task_text = sysp.strip() + "\n\n# Task\n" + t["text_prompt"]  # truyền như biến Jinja {{task}}, không cần escape
        instance = INSTANCE.replace("{blender}", a.blender).replace("{check}", check)
        model = get_model(input_model_name=f"openai/{a.model}", config={
            "model_class": "litellm_textbased",
            "model_kwargs": {"api_base": a.base_url, "api_key": "none", "drop_params": True, "temperature": 0.7,
                             "top_p": 0.95, "max_tokens": a.max_tokens, "extra_body": json.loads(a.extra_body)},
            "cost_tracking": "ignore_errors",
            "observation_template": base_cfg["model"]["observation_template"],
            "format_error_template": base_cfg["model"]["format_error_template"],
        })
        env = LocalEnvironment(cwd=str(d), env=base_cfg["environment"]["env"], timeout=600)
        agent = DefaultAgent(model, env, system_template=base_cfg["agent"]["system_template"],
                             instance_template=instance,
                             step_limit=a.step_limit, cost_limit=0, wall_time_limit_seconds=a.wall_limit,
                             output_path=d / "trajectory.json")
        t0 = time.time()
        try:
            res = agent.run(task_text)
        except Exception as e:  # agent hỏng giữa chừng: vẫn chấm model.py nếu có
            res = {"exit_status": f"crash {type(e).__name__}: {str(e)[:200]}"}
        dt = time.time() - t0
        code = (d / "model.py").read_text(encoding="utf-8", errors="replace") if (d / "model.py").exists() else ""
        rec = {"tag": a.tag, "model": a.model, "task": t["id"], "setting": "text", "thinking": "on", "raw": "",
               "code": code, "out_tokens": 0, "finish_reason": res.get("exit_status"), "steps": agent.n_calls,
               "seconds": round(dt, 1), "exit_status": res.get("exit_status")}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{a.tag}] {t['id']}: {res.get('exit_status')} sau {agent.n_calls} bước, {dt:.0f}s, model.py {len(code)} ký tự",
              flush=True)


if __name__ == "__main__":
    main()
