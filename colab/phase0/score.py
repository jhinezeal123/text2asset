"""Mốc 0: tạo manifest render và chấm điểm.

python score.py --work /content/t2a --gt-manifest      # manifest cho code ground-truth
python score.py --work /content/t2a --gen-manifest     # ghi code sinh ra thành file + manifest
python score.py --work /content/t2a --score            # chấm & viết results/summary.{json,md}

Chỉ số (so với render GT bằng CÙNG renderer): exec rate, mask IoU 4 góc, DINOv2 cosine 4 góc,
Chamfer (điểm bề mặt chuẩn hóa về hộp [-1,1]^3). Mẫu lỗi tính IoU = DINO = 0 ở cột *_all.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

VIEWS = ["view_045.png", "view_135.png", "view_225.png", "view_315.png"]


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def gt_manifest(w):
    with open(os.path.join(w, "manifest_gt.jsonl"), "w") as f:
        for t in jl(os.path.join(w, "tasks.jsonl")):
            f.write(json.dumps({"code_path": t["gt_code"], "out_dir": os.path.join(w, "render", "gt", t["id"])}) + "\n")


def gen_manifest(w):
    n = 0
    with open(os.path.join(w, "manifest_gen.jsonl"), "w") as f:
        for gp in sorted(glob.glob(os.path.join(w, "gen", "*.jsonl"))):
            for r in jl(gp):
                grp = f"{r['setting']}_{r['thinking']}"
                cp = os.path.join(w, "gencode", r["tag"], grp, f"{r['task']}.py")
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                with open(cp, "w", encoding="utf-8") as g:
                    g.write(r["code"])
                f.write(json.dumps({"code_path": cp, "out_dir": os.path.join(w, "render", r["tag"], grp, r["task"])}) + "\n")
                n += 1
    print(f"manifest_gen: {n} job")


class Dino:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.dev).eval()
        self.cache = {}

    def emb(self, path):
        if path not in self.cache:
            from PIL import Image
            im = Image.open(path).convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, im).convert("RGB")
            with self.torch.no_grad():
                x = self.proc(images=rgb, return_tensors="pt").to(self.dev)
                e = self.model(**x).pooler_output[0].float().cpu().numpy()
            self.cache[path] = e / (np.linalg.norm(e) + 1e-8)
        return self.cache[path]


def mask(path):
    from PIL import Image
    return np.array(Image.open(path).convert("RGBA"))[..., 3] > 127


def chamfer(p_dir, q_dir):
    from scipy.spatial import cKDTree
    pa, qa = os.path.join(p_dir, "points.npy"), os.path.join(q_dir, "points.npy")
    if not (os.path.exists(pa) and os.path.exists(qa)):
        return None
    P, Q = np.load(pa), np.load(qa)
    return float((cKDTree(Q).query(P)[0].mean() + cKDTree(P).query(Q)[0].mean()) / 2)


def compare(gen_dir, gt_dir, dino):
    out = {"iou": None, "dino": None, "chamfer": chamfer(gen_dir, gt_dir)}
    pairs = [(os.path.join(gen_dir, v), os.path.join(gt_dir, v)) for v in VIEWS]
    if all(os.path.exists(a) and os.path.exists(b) for a, b in pairs):
        ious, sims = [], []
        for a, b in pairs:
            ma, mb = mask(a), mask(b)
            union = (ma | mb).sum()
            ious.append(float((ma & mb).sum() / union) if union else 0.0)
            sims.append(float(dino.emb(a) @ dino.emb(b)))
        out["iou"], out["dino"] = float(np.mean(ious)), float(np.mean(sims))
    return out


def load_result(d):
    p = os.path.join(d, "result.json")
    return json.load(open(p)) if os.path.exists(p) else {"ok": False, "error": "MISSING"}


def err_kind(e):
    if not e:
        return "OK"
    return e.split(":")[0][:60]


def make_sheet(w, rows, tags_groups, setting, out_png, n=8):
    from PIL import Image, ImageDraw
    cell = 128
    tasks = [t["id"] for t in jl(os.path.join(w, "tasks.jsonl"))][:n]
    cols = [("GT", os.path.join(w, "render", "gt"))] + [
        (f"{tag}/{g.split('_')[1]}", os.path.join(w, "render", tag, g)) for tag, g in tags_groups if g.startswith(setting)]
    sheet = Image.new("RGB", (cell * len(cols), cell * len(tasks) + 20), "white")
    dr = ImageDraw.Draw(sheet)
    for ci, (name, base) in enumerate(cols):
        dr.text((ci * cell + 2, 2), name[:20], fill="black")
        for ri, tid in enumerate(tasks):
            pth = os.path.join(base, tid, VIEWS[0])
            if os.path.exists(pth):
                im = Image.open(pth).convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im).convert("RGB").resize((cell, cell))
                sheet.paste(im, (ci * cell, 20 + ri * cell))
            else:
                dr.text((ci * cell + 30, 20 + ri * cell + 60), "FAIL", fill="red")
    sheet.save(out_png)


def score(w):
    res_dir = os.path.join(w, "results")
    os.makedirs(res_dir, exist_ok=True)
    tasks = jl(os.path.join(w, "tasks.jsonl"))
    summary = {}

    # --- ground truth
    gt = {t["id"]: load_result(os.path.join(w, "render", "gt", t["id"])) for t in tasks}
    summary["gt_exec_rate"] = round(np.mean([r["ok"] for r in gt.values()]), 3)
    summary["gt_failures"] = {k: v.get("error") for k, v in gt.items() if not v["ok"]}

    # --- tập exec 3DCodeData
    man = os.path.join(w, "manifest_execset.jsonl")
    if os.path.exists(man):
        ex = collections.defaultdict(list)
        for j in jl(man):
            r = load_result(j["out_dir"])
            ex[j["kind"]].append((j["factory"], r))
        summary["execset"] = {}
        for kind, lst in ex.items():
            oks = [r["ok"] for _, r in lst]
            summary["execset"][kind] = {
                "n": len(lst), "exec_rate": round(float(np.mean(oks)), 3),
                "timeout_rate": round(float(np.mean([r.get("error") == "TIMEOUT" for _, r in lst])), 3),
                "median_exec_s": float(np.median([r["exec_s"] for _, r in lst if r.get("exec_s") is not None] or [0])),
                "median_tris_ok": int(np.median([r["tris"] for _, r in lst if r["ok"]] or [0])),
                "errors": collections.Counter(err_kind(r.get("error")) for _, r in lst if not r["ok"]).most_common(10),
                "failed_categories": collections.Counter(f.rsplit("_", 1)[0] for f, r in lst if not r["ok"]).most_common(15),
            }

    # --- model zero-shot
    dino = Dino()
    rows = []
    for gp in sorted(glob.glob(os.path.join(w, "gen", "*.jsonl"))):
        for r in jl(gp):
            grp = f"{r['setting']}_{r['thinking']}"
            d = os.path.join(w, "render", r["tag"], grp, r["task"])
            er = load_result(d)
            row = {"tag": r["tag"], "group": grp, "task": r["task"], "ok": er["ok"], "error": er.get("error"),
                   "out_tokens": r["out_tokens"], "truncated": r["finish_reason"] == "length", "tris": er.get("tris", 0)}
            if er["ok"] and gt[r["task"]]["ok"]:
                row.update(compare(d, os.path.join(w, "render", "gt", r["task"]), dino))
            rows.append(row)
    with open(os.path.join(res_dir, "per_sample.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    agg = []
    by = collections.defaultdict(list)
    for row in rows:
        by[(row["tag"], row["group"])].append(row)
    for (tag, grp), lst in sorted(by.items()):
        def m(key, fail_zero):
            vals = [(x.get(key) if x.get(key) is not None else (0.0 if fail_zero else None)) for x in lst]
            vals = [v for v in vals if v is not None]
            return round(float(np.mean(vals)), 3) if vals else None
        ch = [x["chamfer"] for x in lst if x.get("chamfer") is not None]
        agg.append({"tag": tag, "group": grp, "n": len(lst),
                    "exec_rate": round(float(np.mean([x["ok"] for x in lst])), 3),
                    "iou_all": m("iou", True), "dino_all": m("dino", True),
                    "iou_ok": m("iou", False), "dino_ok": m("dino", False),
                    "chamfer_ok_median": round(float(np.median(ch)), 4) if ch else None,
                    "mean_out_tokens": int(np.mean([x["out_tokens"] for x in lst])),
                    "truncated_rate": round(float(np.mean([x["truncated"] for x in lst])), 3),
                    "top_errors": collections.Counter(err_kind(x["error"]) for x in lst if not x["ok"]).most_common(3)})
    summary["models"] = agg
    tok = os.path.join(res_dir, "token_stats.json")
    if os.path.exists(tok):
        summary["token_stats"] = json.load(open(tok))
    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tg = sorted({(a["tag"], a["group"]) for a in agg})
    for s in ("image", "text"):
        try:
            make_sheet(w, rows, tg, s, os.path.join(res_dir, f"sheet_{s}.png"))
        except Exception as e:
            print("sheet error", s, e)

    L = ["# Mốc 0 — kết quả zero-shot", "",
         f"- Tác vụ benchmark: {len(tasks)} | GT chạy được (renderer của mình): {summary['gt_exec_rate']}", ""]
    if "execset" in summary:
        L += ["## Tỷ lệ chạy được của 3DCodeData trên Blender 5.0", "",
              "| bản | n | chạy được | timeout | exec_s trung vị | tris trung vị |", "|---|---|---|---|---|---|"]
        for k, v in summary["execset"].items():
            L.append(f"| {k} | {v['n']} | {v['exec_rate']} | {v['timeout_rate']} | {v['median_exec_s']} | {v['median_tris_ok']} |")
        L.append("")
    if "token_stats" in summary:
        ts = summary["token_stats"]
        L += ["## Độ dài token (tokenizer " + ts["tokenizer"] + ")", "",
              "| bản | p50 | p90 | ≤8K | ≤16K | ≤32K |", "|---|---|---|---|---|---|"]
        for k in ("code", "code_geo"):
            v = ts[k]
            L.append(f"| {k} | {v['p50']} | {v['p90']} | {v['frac_le_8k']} | {v['frac_le_16k']} | {v['frac_le_32k']} |")
        L.append("")
    L += ["## Model zero-shot", "", "| model | setting_thinking | exec | IoU(all) | DINO(all) | IoU(ok) | DINO(ok) | Chamfer(ok, med) | out tok | cắt cụt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in sorted(agg, key=lambda x: (x["group"], -(x["dino_all"] or 0))):
        L.append(f"| {a['tag']} | {a['group']} | {a['exec_rate']} | {a['iou_all']} | {a['dino_all']} | {a['iou_ok']} | "
                 f"{a['dino_ok']} | {a['chamfer_ok_median']} | {a['mean_out_tokens']} | {a['truncated_rate']} |")
    with open(os.path.join(res_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--gt-manifest", action="store_true")
    p.add_argument("--gen-manifest", action="store_true")
    p.add_argument("--score", action="store_true")
    a = p.parse_args()
    if a.gt_manifest:
        gt_manifest(a.work)
    if a.gen_manifest:
        gen_manifest(a.work)
    if a.score:
        score(a.work)


if __name__ == "__main__":
    main()
