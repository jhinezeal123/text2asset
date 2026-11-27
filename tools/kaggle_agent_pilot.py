"""Pilot trên Kaggle GPU (T4 x2): Inkling-2-Mini SQ-20 qua Strata, chạy trong mini-swe-agent, rồi render + chấm điểm.

python -I -X utf8 tools/kaggle_agent_pilot.py --tasks Auger_seed0 --out results/kaggle_agent_pilot
Kết quả tải về <out>/out/: pilot.json (bước, token, tốc độ), trajectory, per_sample.jsonl, compare_sheet.png, log engine.
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["blender_exec.py", "exec_runner.py", "score.py", "data_prep.py", "zeroshot_gen.py", "agent_run.py"]

REMOTE = r'''
import base64, io, json, os, shutil, subprocess, sys, tarfile, time, urllib.request
P = "/tmp/pl"; W = "/tmp/t2a"; OUT = "/kaggle/working/out"; PORT = 8090
os.makedirs(P, exist_ok=True); os.makedirs(OUT, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(P)
cfg = json.load(open(f"{P}/cfg.json")); C = f"{P}/code"; T0 = time.time(); info = {"cfg": cfg}
def sh(c, check=True, quiet=False):
    print(f"[{time.time() - T0:6.0f}s] >> {c[:160]}", flush=True)
    r = subprocess.run(c, shell=True, executable="/bin/bash", capture_output=quiet, text=True)
    if check and r.returncode:
        raise SystemExit(f"lỗi rc={r.returncode}: {c[:200]}")
    return r
def save():
    json.dump(info, open(f"{OUT}/pilot.json", "w"), indent=2, ensure_ascii=False)
try:
    sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2; df -h / /tmp /kaggle/tmp 2>/dev/null")
    # Blender (nền) + scipy cho Python của Blender
    bl = subprocess.Popen(["bash", "-c", """set -e; V=$(curl -s https://download.blender.org/release/Blender5.0/ | grep -oE 'blender-5\\.0\\.[0-9]+-linux-x64\\.tar\\.xz' | sort -uV | tail -1)
curl -sL https://download.blender.org/release/Blender5.0/$V -o /tmp/b.tar.xz && mkdir -p /tmp/blender && tar -xf /tmp/b.tar.xz -C /tmp/blender --strip-components=1 && rm /tmp/b.tar.xz
apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libegl1 >/dev/null 2>&1 || true
BPY=$(ls /tmp/blender/*/python/bin/python3.* | grep -v config | head -1)
SP=$($BPY -c 'import site; print(site.getsitepackages()[0])'); PV=$($BPY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
pip install -q --no-deps --only-binary=:all: --python-version $PV --target $SP 'scipy<1.16' >/dev/null 2>&1 || true
echo BLENDER_READY"""], stdout=open("/tmp/blender_setup.log", "w"), stderr=subprocess.STDOUT)
    sh("pip install -q mini-swe-agent pyyaml requests huggingface_hub pillow scipy 2>&1 | tail -2", check=False)
    sh(f"python {C}/data_prep.py --work {W} --bench-only --only {','.join(cfg['tasks'])}")
    # Strata + Inkling SQ-20 trên 2 GPU
    data_dir = "/kaggle/tmp/strata" if os.path.isdir("/kaggle/tmp") else "/tmp/strata"
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
    sh("curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o /tmp/si.sh && sh /tmp/si.sh | tail -2")
    t = time.time()
    r = sh(f"strata launch --family inkling-2-mini --model SQ-20 --context {cfg['ctx']} --parallel 1 --vision off --gpu 0,1 "
           f"--data-dir {data_dir} --port {PORT} --no-api-key --yes --no-tui 2>&1 | tail -25", check=False)
    info["strata_launch_s"] = round(time.time() - t)
    for i in range(120):
        try:
            mid = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read())["data"][0]["id"]; break
        except Exception:
            time.sleep(10)
    else:
        raise SystemExit("server Strata không lên")
    eng = sh("cat ~/.local/state/strata/launch/*.engine.log 2>/dev/null | grep -E 'engine started|Loaded|Host KV|memory:|tier|split|device|GPU' | tail -15", quiet=True).stdout
    print(eng); info["engine"] = eng; info["model_id"] = mid
    info["gpu_mem"] = sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", quiet=True).stdout
    save()
    # chờ Blender rồi chạy agent
    bl.wait(); sh("tail -2 /tmp/blender_setup.log; /tmp/blender/blender -b --factory-startup --version | head -1")
    t = time.time()
    sh(f"python {C}/agent_run.py --base-url http://127.0.0.1:{PORT}/v1 --model {mid} --tag inkling_agent --work {W} "
       f"--blender /tmp/blender/blender --tasks {','.join(cfg['tasks'])} --step-limit {cfg['steps']} --wall-limit {cfg['wall']}", check=False)
    info["agent_s"] = round(time.time() - t)
    try:
        m = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read())
        reqs = m.get("requests", [])
        out_tok = sum(x.get("output_tokens") or 0 for x in reqs); dur = sum(x.get("duration_s") or 0 for x in reqs)
        info["requests"] = len(reqs); info["output_tokens"] = out_tok; info["prompt_tokens"] = sum(x.get("prompt_tokens") or 0 for x in reqs)
        info["decode_tok_s"] = round(out_tok / max(dur, 1), 1)
        info["per_request"] = [{k: x.get(k) for k in ("prompt_tokens", "output_tokens", "duration_s", "finish")} for x in reqs]
    except Exception as e:
        info["metrics_error"] = str(e)
    save()
    sh("strata launch --stop", check=False)
    # render + chấm điểm
    sh(f"python {C}/score.py --work {W} --gt-manifest")
    sh(f"python {C}/exec_runner.py --blender /tmp/blender/blender --manifest {W}/manifest_gt.jsonl --render --timeout 300 --threads 2 --label gt", check=False)
    sh(f"python {C}/score.py --work {W} --gen-manifest")
    sh(f"python {C}/exec_runner.py --blender /tmp/blender/blender --manifest {W}/manifest_gen.jsonl --render --timeout 240 --threads 2 --label gen", check=False)
    sh(f"python {C}/score.py --work {W} --score", check=False)
    pj = f"{W}/results/per_sample.jsonl"
    info["scores"] = [json.loads(l) for l in open(pj)] if os.path.exists(pj) else []
except SystemExit as e:
    info["fatal"] = str(e); print("FATAL", e, flush=True)
finally:
    info["total_s"] = round(time.time() - T0); save()
    for src, dst in ((f"{W}/results", "results"), (f"{W}/agent", "agent"), (f"{W}/render", "render"), (f"{W}/gen", "gen")):
        if os.path.exists(src):
            shutil.copytree(src, f"{OUT}/{dst}", ignore=shutil.ignore_patterns("*.npy"), dirs_exist_ok=True)
    sh(f"cp ~/.local/state/strata/launch/*.engine.log {OUT}/ 2>/dev/null; true", check=False)
    print("KAGGLE_PILOT_DONE", json.dumps({k: info.get(k) for k in ("agent_s", "requests", "output_tokens", "decode_tok_s", "fatal")}), flush=True)
'''


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="Auger_seed0")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--wall", type=int, default=5400, help="giây tối đa cho agent mỗi đề")
    p.add_argument("--ctx", type=int, default=65536, help="ngữ cảnh: agent tích lũy lịch sử nhiều lượt nên cần dài hơn")
    p.add_argument("--slug", default="t2a-agent-pilot")
    p.add_argument("--out", default="results/kaggle_agent_pilot")
    p.add_argument("--timeout-min", type=float, default=240)
    a = p.parse_args()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name, data):
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        for f in CODE:
            add(f"code/{f}", (ROOT / "colab" / "phase0" / f).read_bytes().replace(b"\r\n", b"\n"))
        add("cfg.json", json.dumps({"tasks": a.tasks.split(","), "steps": a.steps, "wall": a.wall, "ctx": a.ctx}).encode())
    payload = base64.b64encode(buf.getvalue()).decode()

    user = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["username"]
    kid = f"{user}/{a.slug}"
    kdir = ROOT / ".kaggle_kernel_pilot"
    kdir.mkdir(exist_ok=True)
    (kdir / "script.py").write_text(f"PAYLOAD = {payload!r}\n" + REMOTE, encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": a.slug, "code_file": "script.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2), encoding="utf-8")
    r = run(["kaggle", "kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4"])
    print((r.stdout + r.stderr).strip()[-600:], flush=True)
    if r.returncode:
        sys.exit(1)
    t0, status = time.time(), ""
    while time.time() - t0 < a.timeout_min * 60:
        time.sleep(60)
        s = run(["kaggle", "kernels", "status", kid]).stdout.strip()
        status = s.lower()
        print(f"[{time.time() - t0:5.0f}s] KAGGLE_STATUS {s[-60:]}", flush=True)
        if any(k in status for k in ("complete", "error", "cancel")):
            break
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kid, "-p", str(out), "-o"])
    pj = out / "out" / "pilot.json"
    print("KAGGLE_PILOT_END", status[-40:], pj.read_text(encoding="utf-8")[:1500] if pj.exists() else "(không có pilot.json)")


if __name__ == "__main__":
    main()
