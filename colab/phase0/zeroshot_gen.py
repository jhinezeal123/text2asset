"""Sinh code zero-shot bằng vLLM cho một model trên các tác vụ 3DCodeBench (chạy trong venv có vllm).

python zeroshot_gen.py --model Qwen/Qwen3.5-9B --tag qwen35_9b --work /content/t2a --thinking off,on
Dùng đúng system prompt + temperature 0.7 của harness 3DCodeBench. Ghi gen/<tag>.jsonl (bỏ qua mục đã có).
"""

import argparse
import base64
import io
import json
import os
import re
import time

from PIL import Image


def img_url(path, size):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_code(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    text = re.sub(r"^\s*```(?:python|py)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--settings", default="text,image")
    p.add_argument("--thinking", default="off", help="off | on | off,on")
    p.add_argument("--quant", default=None, help="vd fp8 (lượng tử hóa khi nạp)")
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--max-tokens-off", type=int, default=16384)
    p.add_argument("--max-tokens-on", type=int, default=32768)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--gpu-util", type=float, default=0.90)
    a = p.parse_args()

    from vllm import LLM, SamplingParams

    tasks = [json.loads(l) for l in open(os.path.join(a.work, "tasks.jsonl"), encoding="utf-8")]
    sysp = {s: open(os.path.join(a.work, "bench", f"{s}_to_3d_system_prompt.txt"), encoding="utf-8").read()
            for s in ("text", "image")}
    out_path = os.path.join(a.work, "gen", f"{a.tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["task"], r["setting"], r["thinking"]))

    t0 = time.time()
    llm = LLM(model=a.model, quantization=a.quant, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              limit_mm_per_prompt={"image": 4}, enable_prefix_caching=True, seed=0)
    print(f"[{a.tag}] nạp model {time.time() - t0:.0f}s", flush=True)

    for thinking in a.thinking.split(","):
        on = thinking == "on"
        sp = SamplingParams(temperature=0.7, top_p=0.95 if on else 0.8, top_k=20,
                            max_tokens=a.max_tokens_on if on else a.max_tokens_off, seed=0)
        for setting in a.settings.split(","):
            todo = [t for t in tasks if (t["id"], setting, thinking) not in done]
            if not todo:
                continue
            convs = []
            for t in todo:
                if setting == "image":
                    content = [{"type": "image_url", "image_url": {"url": img_url(pth, a.img_size)}} for pth in t["images"]]
                    content.append({"type": "text", "text": t["image_prompt"]})
                else:
                    content = [{"type": "text", "text": t["text_prompt"]}]
                convs.append([{"role": "system", "content": sysp[setting]}, {"role": "user", "content": content}])
            t1 = time.time()
            outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": on}, use_tqdm=False)
            dt = time.time() - t1
            with open(out_path, "a", encoding="utf-8") as f:
                for t, o in zip(todo, outs):
                    c = o.outputs[0]
                    f.write(json.dumps({"tag": a.tag, "model": a.model, "task": t["id"], "setting": setting,
                                        "thinking": thinking, "raw": c.text, "code": extract_code(c.text),
                                        "out_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                                        "prompt_tokens": len(o.prompt_token_ids or [])}, ensure_ascii=False) + "\n")
            ntok = sum(len(o.outputs[0].token_ids) for o in outs)
            print(f"[{a.tag}] {setting}/thinking={thinking}: {len(todo)} mẫu, {dt:.0f}s, {ntok / max(dt, 1):.0f} tok/s",
                  flush=True)


if __name__ == "__main__":
    main()
