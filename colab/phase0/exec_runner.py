"""Chạy song song nhiều script Blender theo manifest (jsonl: {"code_path", "out_dir"}), có timeout cứng.

python exec_runner.py --blender /content/blender/blender --manifest m.jsonl [--render] [--timeout 120]
Bỏ qua job đã có result.json (chạy lại được giữa chừng).
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(job, a):
    out = job["out_dir"]
    rj = os.path.join(out, "result.json")
    if os.path.exists(rj):
        return "skip"
    os.makedirs(out, exist_ok=True)
    cmd = [a.blender, "-b", "--factory-startup", "--python", os.path.join(HERE, "blender_exec.py"), "--",
           job["code_path"], out, "--res", str(a.res), "--samples", str(a.samples), "--threads", str(a.threads)]
    if a.render:
        cmd.append("--render")
    t = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=a.timeout,
                           text=True, errors="replace")
        if not os.path.exists(rj):
            with open(rj, "w") as f:
                json.dump({"ok": False, "error": f"NO_RESULT rc={p.returncode}", "log_tail": p.stdout[-3000:]}, f)
    except subprocess.TimeoutExpired:
        with open(rj, "w") as f:
            json.dump({"ok": False, "error": "TIMEOUT", "wall_s": round(time.time() - t, 1)}, f)
    return "done"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blender", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--render", action="store_true")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--threads", type=int, default=2, help="luồng CPU mỗi tiến trình Blender")
    p.add_argument("--workers", type=int, default=0, help="0 = nproc // threads")
    p.add_argument("--label", default="exec")
    a = p.parse_args()

    jobs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    workers = a.workers or max(1, (os.cpu_count() or 2) // a.threads)
    print(f"[{a.label}] {len(jobs)} job, {workers} worker, timeout {a.timeout}s", flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_one, j, a) for j in jobs]
        for f in as_completed(futs):
            n += 1
            if n % max(1, len(jobs) // 10) == 0 or n == len(jobs):
                print(f"[{a.label}] {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
