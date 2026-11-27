"""Chạy thử LoRA Gemma 4 E4B (ảnh + chữ -> code Blender) bằng Tunix trên 1 chip TPU (v6e-1).

python train_smoke_e4b.py --work /content/t2a_train --n-examples 48 --steps 20 --max-seq 8192
Mục tiêu: xác nhận nạp được model + LoRA + ảnh, đo bộ nhớ HBM, thời gian mỗi bước, loss trước/sau. Ghi smoke_result.json.
Dữ liệu: 3DCodeData (4 ảnh render + code_geo), prompt image_to_3d của 3DCodeBench, loss chỉ tính trên phần code.
"""

import argparse
import dataclasses
import json
import os
import random
import time
import traceback
import urllib.request

import numpy as np

REPO = "YipengGao/3DCodeBench"
GH_RAW = "https://raw.githubusercontent.com/gaoypeng/3dcodebench/main/prompts/"
IMG_NAMES = ["Image_005.webp", "Image_015.webp", "Image_025.webp", "Image_035.webp"]
USER_PROMPT = ("Reconstruct the object shown in the following reference image(s) as a Blender 5.0 Python script. "
               "Treat all images as views of the SAME object.")
LORA_TARGETS = r".*layers.*(q_einsum|k_einsum|kv_einsum|attn_vec_einsum|gate_proj|up_proj|down_proj).*"
RES = {"stages": {}}


def log(msg):
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage(name, t0):
    RES["stages"][name] = round(time.time() - t0, 1)
    log(f"STAGE_DONE {name} {RES['stages'][name]}s")


def hbm():
    import jax
    out = []
    for d in jax.local_devices():
        s = d.memory_stats() or {}
        out.append({"in_use_gb": round(s.get("bytes_in_use", 0) / 2**30, 2),
                    "peak_gb": round(s.get("peak_bytes_in_use", 0) / 2**30, 2),
                    "limit_gb": round(s.get("bytes_limit", 0) / 2**30, 2)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/content/t2a_train")
    p.add_argument("--model-dir", default="/content/gemma-4-E4B-it")
    p.add_argument("--n-examples", type=int, default=48)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--remat", default="DECODER", help="NONE | BLOCK | DECODER")
    p.add_argument("--batch", type=int, default=1, help="mẫu mỗi bước (v5e-8 với FSDP 8 chip: 8)")
    p.add_argument("--mesh", default="1,1", help="kích thước (fsdp,tp), vd 1,1 cho v6e-1; 8,1 cho v5e-8")
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    RES["args"] = vars(a)
    out_json = os.path.join(a.work, "smoke_result.json")
    try:
        run(a)
    except Exception as e:
        RES["error"] = f"{type(e).__name__}: {e}"[:2000]
        RES["traceback"] = traceback.format_exc()[-6000:]
        log("SMOKE_FAILED " + RES["error"][:300])
        print(RES["traceback"], flush=True)
    finally:
        json.dump(RES, open(out_json, "w"), indent=2, ensure_ascii=False)
        log(f"kết quả -> {out_json}")


def run(a):
    t = time.time()
    import jax
    import jax.numpy as jnp
    log(f"jax {jax.__version__}, thiết bị: {jax.devices()}")
    RES["devices"] = [str(d) for d in jax.devices()]

    # ---------- dữ liệu ----------
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image
    pq = hf_hub_download(REPO, "3DCodeData/data/train.parquet", repo_type="dataset")
    df = pd.read_parquet(pq, columns=["factory", "code_geo"]).dropna()
    rows = df.sample(n=min(a.n_examples * 2, len(df)), random_state=0).to_dict("records")
    sysp = urllib.request.urlopen(GH_RAW + "image_to_3d_system_prompt.txt").read().decode()
    data_dir = os.path.join(a.work, "data")
    examples = []
    for r in rows:
        if len(examples) >= a.n_examples:
            break
        try:
            paths = [hf_hub_download(REPO, f"3DCodeData/{r['factory']}/renders/{n}", repo_type="dataset", local_dir=data_dir)
                     for n in IMG_NAMES]
        except Exception as e:
            log(f"bỏ {r['factory']}: {type(e).__name__}")
            continue
        examples.append({"id": r["factory"], "code": r["code_geo"], "images": paths})
    stage("data", t)
    log(f"{len(examples)} mẫu, ví dụ {examples[0]['id']}")

    # ---------- model ----------
    t = time.time()
    snapshot_download("google/gemma-4-E4B-it", local_dir=a.model_dir)
    stage("download_model", t)
    t = time.time()
    import optax
    import qwix
    from flax import nnx
    from transformers import AutoProcessor, AutoTokenizer
    from tunix.models.gemma4 import config as g4cfg
    from tunix.models.gemma4 import model as g4
    from tunix.models.gemma4 import params_safetensors
    from tunix.processors import image_processor
    from tunix.sft import metrics_logger, peft_trainer
    from tunix.sft import utils as sft_utils

    mesh_shape = tuple(int(x) for x in a.mesh.split(","))
    mesh = jax.make_mesh(mesh_shape, ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    RES["mesh"] = mesh_shape
    cfg = g4.ModelConfig.gemma4_e4b()
    cfg = dataclasses.replace(cfg, remat_config=getattr(g4cfg.RematConfig, a.remat))
    model = params_safetensors.create_model_from_safe_tensors(a.model_dir, cfg, mesh, dtype=jnp.bfloat16, text_only=False)
    stage("load_model", t)
    RES["hbm_after_load"] = hbm()
    log(f"HBM sau khi nạp: {RES['hbm_after_load']}")

    t = time.time()
    lora = qwix.LoraProvider(module_path=LORA_TARGETS, rank=a.rank, alpha=a.alpha)
    model = qwix.apply_lora_to_model(model, lora, **model.get_model_input())
    lora_state = nnx.state(model, nnx.LoRAParam)
    n_lora = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(lora_state))
    RES["lora_params_m"] = round(n_lora / 1e6, 2)
    stage("apply_lora", t)
    log(f"LoRA: {RES['lora_params_m']}M tham số")

    # ---------- tokenize + ảnh ----------
    t = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    proc = AutoProcessor.from_pretrained(a.model_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    items, lens, skipped = [], [], 0
    for ex in examples:
        msgs = [{"role": "system", "content": sysp},
                {"role": "user", "content": [{"type": "image"}] * 4 + [{"type": "text", "text": USER_PROMPT}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = proc.apply_chat_template(msgs + [{"role": "assistant", "content": f"```python\n{ex['code']}\n```"}],
                                        tokenize=False, enable_thinking=False)
        if not full.startswith(prompt):  # phòng khi template chèn gì đó khác
            full = prompt + f"```python\n{ex['code']}\n```" + "<turn|>"
        src = np.array(tok(prompt, add_special_tokens=False)["input_ids"], dtype=np.int32)
        dst = np.array(tok(full[len(prompt):], add_special_tokens=False)["input_ids"], dtype=np.int32)
        imgs = [Image.open(pth).convert("RGB") for pth in ex["images"]]
        _, toks = image_processor.process_gemma4_inputs([imgs], [src], model.vision_encoder, pad_id)  # chỉ để đo độ dài
        src_x = np.asarray(toks[0], dtype=np.int32)
        L = len(src_x) + len(dst)
        lens.append(L)
        if L > a.max_seq:
            skipped += 1
            continue
        items.append((imgs, src, dst))
    batches = []
    for b0 in range(0, len(items) - a.batch + 1, a.batch):  # gom a.batch mẫu, xử lý ảnh theo batch
        grp = items[b0:b0 + a.batch]
        processed, toks = image_processor.process_gemma4_inputs([g[0] for g in grp], [g[1] for g in grp],
                                                                model.vision_encoder, pad_id)
        ids = np.full((len(grp), a.max_seq), pad_id, np.int32)
        mask = np.zeros((len(grp), a.max_seq), bool)
        for j, (g, sx) in enumerate(zip(grp, toks)):
            sx = np.asarray(sx, dtype=np.int32)
            L = min(len(sx) + len(g[2]), a.max_seq)
            ids[j, :len(sx)] = sx
            ids[j, len(sx):L] = g[2][:L - len(sx)]
            mask[j, len(sx):L] = True
        batches.append(peft_trainer.TrainingInput(input_tokens=ids, input_mask=mask, images=processed))
    RES["seq_len"] = {"p50": int(np.median(lens)), "max": int(max(lens)), "n_fit": len(items), "n_batches": len(batches),
                      "n_skipped": skipped,
                      "prompt_tokens_example": int(len(src_x))}
    stage("tokenize", t)
    log(f"độ dài chuỗi: {RES['seq_len']}")
    if not batches:
        raise RuntimeError("không có mẫu nào vừa max_seq")

    def gen_input(x):
        pad_mask = x.input_tokens != pad_id
        return {"input_tokens": x.input_tokens, "input_mask": x.input_mask,
                "positions": sft_utils.build_positions_from_mask(pad_mask),
                "attention_mask": sft_utils.make_causal_attn_mask(pad_mask), "images": x.images}

    def eval_loss(b):
        with mesh:
            out = peft_trainer._default_loss_fn(model, **gen_input(b))
        return float(out[0] if isinstance(out, tuple) else out)

    t = time.time()
    RES["loss_before"] = eval_loss(batches[0])
    stage("first_forward", t)
    log(f"loss trước khi train (mẫu 0): {RES['loss_before']:.4f}; HBM {hbm()}")

    # ---------- train ----------
    pulls = []

    def train_ds():  # ghi thời điểm lấy từng batch -> suy ra thời gian mỗi bước (bước đầu gồm biên dịch)
        for i in range(a.steps):
            pulls.append(time.time())
            yield batches[i % len(batches)]
    cfg_t = peft_trainer.TrainingConfig(
        eval_every_n_steps=10**9, max_steps=a.steps,
        metrics_logging_options=metrics_logger.MetricsLoggerOptions(log_dir=os.path.join(a.work, "tb"), flush_every_n_steps=1))
    trainer = peft_trainer.PeftTrainer(model, optax.adamw(a.lr), cfg_t).with_gen_model_input_fn(gen_input)
    t = time.time()
    with mesh:
        trainer.train(train_ds(), None)
    dt = time.time() - t
    gaps = np.diff(pulls + [time.time()])
    RES["step_seconds"] = [round(float(g), 2) for g in gaps]
    if len(gaps) > 3:
        RES["sec_per_step_warm"] = round(float(np.median(gaps[2:])), 2)
        RES["tokens_per_sec_warm"] = round(a.batch * a.max_seq / max(RES["sec_per_step_warm"], 1e-6))
    stage("train", t)
    RES["train_seconds"] = round(dt, 1)
    RES["hbm_after_train"] = hbm()
    RES["loss_after"] = eval_loss(batches[0])
    log(f"train {a.steps} bước trong {dt:.0f}s (gồm biên dịch); loss mẫu 0: {RES['loss_before']:.4f} -> {RES['loss_after']:.4f}")
    log(f"HBM sau train: {RES['hbm_after_train']}")

    log(f"thời gian từng bước: {RES.get('step_seconds')}; sau biên dịch ~{RES.get('sec_per_step_warm')}s/bước "
        f"(~{RES.get('tokens_per_sec_warm')} token/s ở chuỗi {a.max_seq}, gồm cả phần đệm)")

    # ---------- lưu adapter ----------
    t = time.time()
    from orbax import checkpoint as ocp
    ad = os.path.join(a.work, "adapter")
    ocp.StandardCheckpointer().save(ad, nnx.state(model, nnx.LoRAParam), force=True)
    stage("save_adapter", t)
    RES["ok"] = True
    log("SMOKE_OK")


if __name__ == "__main__":
    main()
