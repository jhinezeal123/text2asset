"""Chuẩn bị dữ liệu Mốc 0.

1. Chọn N tác vụ 3DCodeBench (seed cố định), tải mô tả, 4 ảnh tham chiếu, code ground-truth, system prompt gốc.
2. Tải 3DCodeData train.parquet: lấy mẫu K script để đo tỷ lệ chạy được; đo phân bố độ dài token.

python data_prep.py --work /content/t2a --n-tasks 50 --n-exec 300 --n-tok 2000
"""

import argparse
import json
import os
import random
import urllib.request

REPO = "YipengGao/3DCodeBench"
GH_RAW = "https://raw.githubusercontent.com/gaoypeng/3dcodebench/main/prompts/"
IMG_NAMES = ["Image_005.png", "Image_015.png", "Image_025.png", "Image_035.png"]
IMAGE_USER_PROMPT = ("Reconstruct the object shown in the following reference image(s) as a Blender 5.0 Python "
                     "script. Treat all images as views of the SAME object.\n\n"
                     "[reference images: Image_005.png, Image_015.png, Image_025.png, Image_035.png]")


def prep_bench(a, api):
    from huggingface_hub import snapshot_download
    bench_dir = os.path.join(a.work, "bench")
    os.makedirs(bench_dir, exist_ok=True)
    tasks_path = os.path.join(a.work, "tasks.jsonl")

    for name in ("text_to_3d_system_prompt.txt", "image_to_3d_system_prompt.txt"):
        dst = os.path.join(bench_dir, name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(GH_RAW + name, dst)

    cats = sorted(e.path.split("/")[-1] for e in api.list_repo_tree(REPO, path_in_repo="3DCodeBench", repo_type="dataset")
                  if e.path.count("/") == 1)
    chosen = sorted(random.Random(0).sample(cats, min(a.n_tasks, len(cats))))
    if a.only:  # chỉ tải/ghi các task cần (vd chấm lại một phần) — vẫn giữ đúng tập chọn theo seed 0
        chosen = [c for c in chosen if c in set(a.only.split(","))]
    print(f"benchmark: {len(cats)} loại, chọn {len(chosen)}", flush=True)

    patterns = []
    for c in chosen:
        patterns += [f"3DCodeBench/{c}/*", f"3DCodeBench_ModelLogs/inputs/{c}/*", f"3DCodeBench_ModelLogs/inputs/{c}/images/*"]
    snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns, local_dir=bench_dir, max_workers=16)

    with open(tasks_path, "w", encoding="utf-8") as f:
        for c in chosen:
            inp = os.path.join(bench_dir, "3DCodeBench_ModelLogs", "inputs", c)
            desc = open(os.path.join(inp, "prompt_description.txt"), encoding="utf-8").read().strip()
            f.write(json.dumps({
                "id": c,
                "category": c.rsplit("_seed", 1)[0],
                "text_prompt": desc,
                "image_prompt": IMAGE_USER_PROMPT,
                "images": [os.path.join(inp, "images", n) for n in IMG_NAMES],
                "gt_code": os.path.join(bench_dir, "3DCodeBench", c, f"{c}.py"),
            }, ensure_ascii=False) + "\n")
    print(f"tasks -> {tasks_path}", flush=True)


def prep_corpus(a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    pq = hf_hub_download(REPO, "3DCodeData/data/train.parquet", repo_type="dataset",
                         local_dir=os.path.join(a.work, "corpus"))
    df = pd.read_parquet(pq, columns=["factory", "code", "code_geo"])
    df["category"] = df["factory"].str.rsplit("_", n=1).str[0]
    print(f"3DCodeData: {len(df)} dòng, {df.category.nunique()} loại", flush=True)

    # --- mẫu để đo tỷ lệ chạy được (cả bản đầy đủ và bản _geo)
    ex = df.sample(n=min(a.n_exec, len(df)), random_state=0)
    exec_dir = os.path.join(a.work, "execset")
    man = os.path.join(a.work, "manifest_execset.jsonl")
    with open(man, "w") as f:
        for _, r in ex.iterrows():
            for kind in ("code", "code_geo"):
                d = os.path.join(exec_dir, kind, r.factory)
                os.makedirs(d, exist_ok=True)
                cp = os.path.join(d, "script.py")
                with open(cp, "w", encoding="utf-8") as g:
                    g.write(r[kind])
                f.write(json.dumps({"code_path": cp, "out_dir": os.path.join(d, "out"),
                                    "factory": r.factory, "kind": kind}) + "\n")
    print(f"execset: {len(ex)} factory x 2 -> {man}", flush=True)

    # --- phân bố độ dài token (tokenizer của student)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sm = df.sample(n=min(a.n_tok, len(df)), random_state=1)
    stats = {"tokenizer": a.tokenizer, "n": len(sm)}
    import numpy as np
    for kind in ("code", "code_geo"):
        lens = np.array([len(x) for x in tok(sm[kind].tolist(), add_special_tokens=False)["input_ids"]])
        stats[kind] = {
            "p10": int(np.percentile(lens, 10)), "p50": int(np.percentile(lens, 50)),
            "p90": int(np.percentile(lens, 90)), "max": int(lens.max()),
            **{f"frac_le_{k // 1000}k": round(float((lens <= k).mean()), 3) for k in (4096, 8192, 12288, 16384, 32768)},
        }
        sm[f"tok_{kind}"] = lens
    by_cat = sm.groupby("category")["tok_code_geo"].median().sort_values()
    stats["shortest_categories_geo_median"] = {k: int(v) for k, v in by_cat.head(15).items()}
    stats["longest_categories_geo_median"] = {k: int(v) for k, v in by_cat.tail(15).items()}
    os.makedirs(os.path.join(a.work, "results"), exist_ok=True)
    with open(os.path.join(a.work, "results", "token_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("token stats:", json.dumps({k: stats[k] for k in ("code", "code_geo")}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--n-tasks", type=int, default=50)
    p.add_argument("--n-exec", type=int, default=300)
    p.add_argument("--n-tok", type=int, default=2000)
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    p.add_argument("--only", default="", help="danh sách task id (phẩy) — chỉ chuẩn bị các task này")
    p.add_argument("--bench-only", action="store_true", help="bỏ qua 3DCodeData (execset, token stats)")
    a = p.parse_args()
    from huggingface_hub import HfApi
    prep_bench(a, HfApi())
    if not a.bench_only:
        prep_corpus(a)


if __name__ == "__main__":
    main()
