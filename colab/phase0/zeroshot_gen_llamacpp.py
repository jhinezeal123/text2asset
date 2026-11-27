"""Sinh code zero-shot qua llama-server (llama.cpp) — dùng cho GGUF như Qwen3.8-27B NariQ 3-bit.

python zeroshot_gen_llamacpp.py --server /content/nari/bin/.../llama-server --model m.gguf --mmproj mm.gguf \
    --tag qwen38_27b_nariq3 --work /content/t2a --thinking off,on
Cùng system prompt, temperature, giới hạn token và định dạng output (gen/<tag>.jsonl) như zeroshot_gen.py.
Ghi từng mẫu ngay khi xong (chạy lại được giữa chừng).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_gen import extract_code, img_url  # noqa: E402


def start_server(a, parallel):
    ctx = parallel * a.slot_ctx
    cmd = [a.server, "-m", a.model, "--mmproj", a.mmproj, "-ngl", "999", "-c", str(ctx), "-np", str(parallel),
           "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--host", "127.0.0.1", "--port", str(a.port),
           "--no-webui"]
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(a.server) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    log = open(os.path.join(a.work, f"llama_server_{a.tag}.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            return None
        try:
            if requests.get(f"http://127.0.0.1:{a.port}/health", timeout=5).status_code == 200:
                used = int(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                                          shell=True, capture_output=True, text=True).stdout.split()[0] or 0)
                if used < 8000:  # model 10 GB phải nằm trên GPU; ít hơn = đang chạy CPU
                    proc.kill()
                    sys.exit(f"[{a.tag}] llama-server chỉ dùng {used} MiB GPU -> đang chạy CPU, dừng")
                print(f"[{a.tag}] llama-server sẵn sàng ({parallel} slot x {a.slot_ctx} ctx) sau {time.time() - t0:.0f}s",
                      flush=True)
                return proc
        except requests.RequestException:
            pass
        time.sleep(5)
    proc.kill()
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="", help="llama-server (bỏ trống khi dùng --base-url)")
    p.add_argument("--model", required=True, help="file GGUF, hoặc tên model khi dùng --base-url")
    p.add_argument("--mmproj", default="")
    p.add_argument("--base-url", default="", help="server OpenAI-compatible có sẵn (vd Strata), không tự khởi động")
    p.add_argument("--api-key", default="")
    p.add_argument("--extra-body", default="", help='JSON gộp vào mỗi request, vd {"reasoning_effort":"high"}')
    p.add_argument("--no-template-kwargs", action="store_true", help="không gửi chat_template_kwargs.enable_thinking")
    p.add_argument("--tag", required=True)
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--settings", default="text,image")
    p.add_argument("--thinking", default="off,on")
    p.add_argument("--parallel", default="8,4", help="thử lần lượt số slot song song cho tới khi server lên")
    p.add_argument("--slot-ctx", type=int, default=40960)
    p.add_argument("--max-tokens-off", type=int, default=16384)
    p.add_argument("--max-tokens-on", type=int, default=32768)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--port", type=int, default=8090)
    a = p.parse_args()

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

    proc, parallel = None, 0
    extra = json.loads(a.extra_body) if a.extra_body else {}
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    if a.base_url:  # server ngoài đã chạy sẵn
        parallel = int(a.parallel.split(",")[0])
        url = a.base_url.rstrip("/") + "/chat/completions"
    for n in ([] if a.base_url else [int(x) for x in a.parallel.split(",")]):
        proc = start_server(a, n)
        if proc:
            parallel = n
            break
        print(f"[{a.tag}] llama-server không lên với {n} slot, thử ít hơn", flush=True)
    if not proc and not a.base_url:
        sys.exit(f"[{a.tag}] không khởi động được llama-server (xem llama_server_{a.tag}.log)")
    if not a.base_url:
        url = f"http://127.0.0.1:{a.port}/v1/chat/completions"

    lock = threading.Lock()

    def one(t, setting, thinking):
        on = thinking == "on"
        if setting == "image":
            content = [{"type": "image_url", "image_url": {"url": img_url(pth, a.img_size)}} for pth in t["images"]]
            content.append({"type": "text", "text": t["image_prompt"]})
        else:
            content = [{"type": "text", "text": t["text_prompt"]}]
        body = {"messages": [{"role": "system", "content": sysp[setting]}, {"role": "user", "content": content}],
                "temperature": 0.7, "top_p": 0.95 if on else 0.8, "top_k": 20, "seed": 0,
                "max_tokens": a.max_tokens_on if on else a.max_tokens_off,
                **({} if a.no_template_kwargs else {"chat_template_kwargs": {"enable_thinking": on}}), **extra}
        if a.base_url:
            body["model"] = a.model
        r = requests.post(url, json=body, headers=headers, timeout=7200)
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        msg = ch["message"]
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        raw = (f"<think>{reasoning}</think>" if reasoning else "") + text
        rec = {"tag": a.tag, "model": os.path.basename(a.model), "task": t["id"], "setting": setting, "thinking": thinking,
               "raw": raw, "code": extract_code(raw), "out_tokens": j.get("usage", {}).get("completion_tokens", 0),
               "finish_reason": ch.get("finish_reason"), "prompt_tokens": j.get("usage", {}).get("prompt_tokens", 0)}
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["out_tokens"]

    try:
        for thinking in a.thinking.split(","):
            for setting in a.settings.split(","):
                todo = [t for t in tasks if (t["id"], setting, thinking) not in done]
                if not todo:
                    continue
                t1, ntok, nerr = time.time(), 0, 0
                with ThreadPoolExecutor(parallel) as ex:
                    futs = [ex.submit(one, t, setting, thinking) for t in todo]
                    for k, f in enumerate(as_completed(futs), 1):
                        try:
                            ntok += f.result()
                        except Exception as e:
                            nerr += 1
                            print(f"[{a.tag}] lỗi request: {type(e).__name__}: {str(e)[:200]}", flush=True)
                        if k % 5 == 0 or k == len(futs):  # tiến độ (monitor bắt dạng "[gen] k/N ")
                            print(f"[gen] {k}/{len(futs)} {setting}/thinking={thinking} "
                                  f"{ntok / max(time.time() - t1, 1):.0f} tok/s, {nerr} lỗi", flush=True)
                dt = time.time() - t1
                print(f"[{a.tag}] {setting}/thinking={thinking}: {len(todo) - nerr} mẫu, {dt:.0f}s, "
                      f"{ntok / max(dt, 1):.0f} tok/s" + (f", {nerr} lỗi" if nerr else ""), flush=True)
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
