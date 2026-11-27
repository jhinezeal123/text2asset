"""Chạy thử LoRA Gemma 4 E4B trên Kaggle TPU (mặc định v5e-8) qua Kaggle CLI, rồi tải smoke_result.json về.

python -I -X utf8 tools/kaggle_train_smoke.py --out results/train_smoke_kaggle
Kernel riêng tư; tự tạo venv Python 3.12 bằng uv (Tunix cần >= 3.12), cài jax[tpu] + Tunix, chạy train_smoke_e4b.py
với FSDP trên 8 chip (batch 8, mesh 8,1).
"""

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REMOTE = r'''
import base64, json, os, subprocess, sys, time
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
open("/tmp/train_smoke_e4b.py", "w").write(base64.b64decode(SCRIPT).decode())
T0 = time.time()
def sh(c):
    print(f"[{time.time() - T0:6.0f}s] >> {c[:200]}", flush=True)
    return subprocess.run(c, shell=True, executable="/bin/bash").returncode
sh("python3 --version; nproc; free -g | head -2; ls /dev/accel* /dev/vfio 2>/dev/null | head; env | grep -i tpu | head")
rc = sh("pip install -q uv && uv venv -q -p 3.12 /tmp/venv && "
        "uv pip install -q --python /tmp/venv/bin/python 'jax[tpu]>=0.6.0,!=0.7.2,<0.12' 'git+https://github.com/google/tunix' "
        "qwix transformers huggingface_hub pandas pyarrow pillow")
sh("/tmp/venv/bin/python -c \"import jax; print('jax', jax.__version__, jax.devices())\"")
if rc == 0:
    sh(f"/tmp/venv/bin/python -u /tmp/train_smoke_e4b.py --work /tmp/t2a_train --model-dir /tmp/gemma-4-E4B-it {ARGS} 2>&1 "
       "| grep -v 'it/s]'")
sh(f"cp /tmp/t2a_train/smoke_result.json {OUT}/ 2>/dev/null; true")
print("KAGGLE_SMOKE_DONE", open(f"{OUT}/smoke_result.json").read()[:1500] if os.path.exists(f"{OUT}/smoke_result.json") else "không có kết quả", flush=True)
'''


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--accelerator", default="TpuV5E8")
    p.add_argument("--args", default="--batch 8 --mesh 8,1 --n-examples 72 --steps 20 --max-seq 8192")
    p.add_argument("--slug", default="t2a-gemma4-e4b-lora-smoke")
    p.add_argument("--out", default="results/train_smoke_kaggle")
    p.add_argument("--timeout-min", type=float, default=180)
    a = p.parse_args()

    script = base64.b64encode((ROOT / "colab" / "phase0" / "train_smoke_e4b.py").read_bytes().replace(b"\r\n", b"\n")).decode()
    user = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["username"]
    kid = f"{user}/{a.slug}"
    kdir = ROOT / ".kaggle_kernel_smoke"
    kdir.mkdir(exist_ok=True)
    (kdir / "script.py").write_text(f"SCRIPT = {script!r}\nARGS = {a.args!r}\n" + REMOTE, encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": a.slug, "code_file": "script.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2), encoding="utf-8")
    r = run(["kaggle", "kernels", "push", "-p", str(kdir), "--accelerator", a.accelerator])
    msg = (r.stdout + r.stderr).strip()
    print("KAGGLE_PUSH", msg[-500:], flush=True)
    if r.returncode or "error" in msg.lower():
        sys.exit(1)
    t0, status = time.time(), ""
    while time.time() - t0 < a.timeout_min * 60:
        time.sleep(60)
        status = run(["kaggle", "kernels", "status", kid]).stdout.strip().lower()
        print(f"[{time.time() - t0:5.0f}s] KAGGLE_STATUS {status[-50:]}", flush=True)
        if any(k in status for k in ("complete", "error", "cancel")):
            break
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kid, "-p", str(out), "-o"])
    pj = out / "out" / "smoke_result.json"
    print("KAGGLE_SMOKE_END", status[-40:], pj.read_text(encoding="utf-8")[:1200] if pj.exists() else "(không có smoke_result.json)")


if __name__ == "__main__":
    main()
