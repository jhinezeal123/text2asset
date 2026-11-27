"""Render + chấm điểm + sheet so sánh trên Kaggle (kernel CPU, riêng tư) qua Kaggle CLI.

python -I -X utf8 tools/kaggle_render.py \
    --cols "Gemma 26B=gemma4_26b:text_off,Qwen3.8-27B=qwen38_27b_nariq3:text_off,Inkling SQ-20=inkling2mini_sq20:text_on" \
    --tasks-from inkling2mini_sq20 --out results/kaggle_render
Mã + các dòng gen cần thiết được nhúng (base64) vào script kernel; kernel tự tải Blender 5.0 và benchmark.
Kết quả tải về <out>/: compare_sheet.png, per_sample.jsonl, summary.md, render/ (ảnh).
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
GEN_DIRS = [ROOT / "results" / "phase0_sync" / "gen", ROOT / "results" / "phase0" / "gen"]
CODE = ["blender_exec.py", "exec_runner.py", "score.py", "data_prep.py"]

REMOTE = r'''
import base64, io, json, os, shutil, subprocess, sys, tarfile
P = "/tmp/pl"; W = "/tmp/t2a"; OUT = "/kaggle/working/out"
os.makedirs(P, exist_ok=True); os.makedirs(OUT, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(P)
cfg = json.load(open(f"{P}/cfg.json"))
def sh(c, check=True):
    print(">>", c[:200], flush=True)
    r = subprocess.run(c, shell=True, executable="/bin/bash")
    if check and r.returncode:
        sys.exit(f"lỗi rc={r.returncode}: {c[:200]}")
sh("nproc; free -g | head -2")
sh("""set -e; V=$(curl -s https://download.blender.org/release/Blender5.0/ | grep -oE 'blender-5\\.0\\.[0-9]+-linux-x64\\.tar\\.xz' | sort -uV | tail -1)
curl -sL https://download.blender.org/release/Blender5.0/$V -o /tmp/b.tar.xz && mkdir -p /tmp/blender && tar -xf /tmp/b.tar.xz -C /tmp/blender --strip-components=1 && rm /tmp/b.tar.xz
apt-get -qq install -y libxi6 libxkbcommon0 libsm6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libegl1 >/dev/null 2>&1 || true
/tmp/blender/blender -b --factory-startup --version | head -1""")
# scipy cho Python của Blender (một số script GT cần)
sh("""BPY=$(ls /tmp/blender/*/python/bin/python3.* | grep -v config | head -1)
SP=$($BPY -c 'import site; print(site.getsitepackages()[0])'); PV=$($BPY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
pip install -q --no-deps --only-binary=:all: --python-version $PV --target $SP 'scipy<1.16' 2>&1 | tail -2
$BPY -c 'import scipy; print("blender scipy", scipy.__version__)' || echo 'WARN Blender thiếu scipy'""", check=False)
sh("pip install -q huggingface_hub pillow scipy >/dev/null 2>&1", check=False)
C = f"{P}/code"
sh(f"python {C}/data_prep.py --work {W} --bench-only --only {','.join(cfg['tasks'])}")
os.makedirs(f"{W}/gen", exist_ok=True)
for f in os.listdir(f"{P}/gen"):
    shutil.copy(f"{P}/gen/{f}", f"{W}/gen/{f}")
sh(f"python {C}/score.py --work {W} --gt-manifest")
sh(f"python {C}/exec_runner.py --blender /tmp/blender/blender --manifest {W}/manifest_gt.jsonl --render --timeout 300 --threads 2 --label gt", check=False)
sh(f"python {C}/score.py --work {W} --gen-manifest")
sh(f"python {C}/exec_runner.py --blender /tmp/blender/blender --manifest {W}/manifest_gen.jsonl --render --timeout 240 --threads 2 --label gen", check=False)
sh(f"python {C}/score.py --work {W} --score", check=False)

# --- sheet so sánh: hàng = đề, cột = GT + từng model (2 góc nhìn mỗi cột)
from PIL import Image, ImageDraw, ImageFont
rows = {}
pj = f"{W}/results/per_sample.jsonl"
for l in (open(pj) if os.path.exists(pj) else []):
    r = json.loads(l); rows[(r["tag"], r["group"], r["task"])] = r
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception:
    font = ImageFont.load_default()
cell, lab, head = 150, 170, 40
views = ["view_045.png", "view_225.png"]
cols = [("GT", f"{W}/render/gt", None)] + [(c["label"], f"{W}/render/{c['tag']}/{c['group']}", (c["tag"], c["group"])) for c in cfg["cols"]]
tasks = cfg["tasks"]
S = Image.new("RGB", (lab + cell * 2 * len(cols), head + cell * len(tasks)), "white"); d = ImageDraw.Draw(S)
for ci, (name, _, key) in enumerate(cols):
    x = lab + ci * 2 * cell
    d.text((x + 6, 6), name, fill="black", font=font)
    if key:
        sel = [r for (t, g, _), r in rows.items() if (t, g) == key]
        if sel:
            ok = sum(r["ok"] for r in sel); iou = sum((r.get("iou") or 0) for r in sel) / len(sel)
            d.text((x + 6, 22), f"chạy {ok}/{len(sel)}  IoU {iou:.2f}", fill="black", font=font)
    d.line((x, 0, x, S.height), fill=(200, 200, 200))
for ri, t in enumerate(tasks):
    y = head + ri * cell
    d.line((0, y, S.width, y), fill=(220, 220, 220))
    d.text((6, y + cell // 2 - 8), t.replace("_seed0", ""), fill="black", font=font)
    for ci, (name, base, key) in enumerate(cols):
        x = lab + ci * 2 * cell
        r = rows.get((*key, t)) if key else None
        for vi, v in enumerate(views):
            p = f"{base}/{t}/{v}"
            if os.path.exists(p):
                im = Image.open(p).convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                S.paste(Image.alpha_composite(bg, im).convert("RGB").resize((cell, cell)), (x + vi * cell, y))
            elif vi == 0:
                d.text((x + 10, y + cell // 2 - 8), "LỖI" if key else "GT lỗi", fill="red", font=font)
        if r is not None:
            txt = f"IoU {r['iou']:.2f}" if r.get("iou") is not None else ("chạy được" if r["ok"] else (r.get("error") or "")[:38])
            d.text((x + 4, y + cell - 18), txt, fill="black" if r["ok"] else "red", font=font)
S.save(f"{OUT}/compare_sheet.png")
for f in ("per_sample.jsonl", "summary.md", "summary.json"):
    if os.path.exists(f"{W}/results/{f}"):
        shutil.copy(f"{W}/results/{f}", f"{OUT}/{f}")
shutil.copytree(f"{W}/render", f"{OUT}/render", ignore=shutil.ignore_patterns("*.npy"), dirs_exist_ok=True)
print("KAGGLE_RENDER_DONE", flush=True)
'''


def load_gen(tag):
    for d in GEN_DIRS:
        p = d / f"{tag}.jsonl"
        if p.exists():
            return [json.loads(l) for l in open(p, encoding="utf-8")]
    sys.exit(f"không thấy gen của {tag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cols", required=True, help="'Nhãn=tag:group,...' (group vd text_off, text_on, image_on)")
    p.add_argument("--tasks-from", required=True, help="tag lấy danh sách đề (các đề tag này đã làm)")
    p.add_argument("--slug", default="t2a-render-compare")
    p.add_argument("--out", default="results/kaggle_render")
    p.add_argument("--timeout-min", type=float, default=90)
    a = p.parse_args()

    cols = []
    for item in a.cols.split(","):
        label, _, tg = item.partition("=")
        tag, _, group = tg.partition(":")
        cols.append({"label": label.strip(), "tag": tag.strip(), "group": group.strip()})
    first = next(c for c in cols if c["tag"] == a.tasks_from)
    tasks = sorted({r["task"] for r in load_gen(a.tasks_from) if f"{r['setting']}_{r['thinking']}" == first["group"]})
    print(f"{len(tasks)} đề: {', '.join(t.replace('_seed0', '') for t in tasks)}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        def add(name, data):
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        for f in CODE:
            add(f"code/{f}", (ROOT / "colab" / "phase0" / f).read_bytes().replace(b"\r\n", b"\n"))
        by_tag = {}
        for c in cols:
            keep = [r for r in load_gen(c["tag"]) if r["task"] in tasks and f"{r['setting']}_{r['thinking']}" == c["group"]]
            by_tag.setdefault(c["tag"], []).extend(keep)
        for tag, recs in by_tag.items():
            add(f"gen/{tag}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode())
        add("cfg.json", json.dumps({"cols": cols, "tasks": tasks}, ensure_ascii=False).encode())
    payload = base64.b64encode(buf.getvalue()).decode()

    user = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))["username"]
    kid = f"{user}/{a.slug}"
    kdir = ROOT / ".kaggle_kernel"
    kdir.mkdir(exist_ok=True)
    (kdir / "script.py").write_text(f"PAYLOAD = {payload!r}\n" + REMOTE, encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": a.slug, "code_file": "script.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2), encoding="utf-8")

    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(kdir)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print((r.stdout + r.stderr).strip())
    if r.returncode:
        sys.exit(1)
    t0 = time.time()
    status = ""
    while time.time() - t0 < a.timeout_min * 60:
        time.sleep(30)
        s = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        status = s.strip().split()[-1].strip('"').lower() if s.strip() else ""
        print(f"[{time.time() - t0:5.0f}s] {s.strip()[-80:]}", flush=True)
        if any(k in status for k in ("complete", "error", "cancel")):
            break
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kid, "-p", str(out), "-o"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"KAGGLE_STATUS {status} -> {out}")


if __name__ == "__main__":
    main()
