"""Sinh code có vòng sửa lỗi: sinh -> chạy thử trong Blender -> lỗi thì gửi traceback -> sửa (tối đa N lượt).

python repair_loop.py --base-url http://127.0.0.1:8090/v1 --model inkling-2-mini-sq-20 --tag inkling_fix \
    --work /content/t2a --blender /content/blender/blender --tasks Auger_seed0 --max-rounds 3
Ghi gen/<tag>_r0.jsonl (lượt đầu = zero-shot) và gen/<tag>.jsonl (kết quả cuối, kèm lịch sử các lượt),
cùng định dạng với zeroshot_gen.py nên score.py chấm được cả hai. Ghi từng đề ngay khi xong (chạy lại được).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_gen import extract_code, img_url  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX_PROMPT = ("The script failed when run in Blender 5.0 (background mode, factory settings):\n\n{err}\n\n"
              "Fix the problem and return the COMPLETE corrected script in a single ```python code block.")


def try_exec(blender, code, timeout):
    """Chạy code trong Blender (không render). Trả (ok, thông báo lỗi ngắn gọn cho model)."""
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "s.py")
        open(cp, "w", encoding="utf-8").write(code)
        try:
            p = subprocess.run([blender, "-b", "--factory-startup", "--python", os.path.join(HERE, "blender_exec.py"),
                                "--", cp, d], capture_output=True, text=True, errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT: the script did not finish within {timeout} s. Make it simpler/faster."
        rj = os.path.join(d, "result.json")
        if not os.path.exists(rj):
            return False, "Blender crashed:\n" + p.stdout[-1500:]
        r = json.load(open(rj))
        if r.get("ok"):
            return True, ""
        if r.get("error"):
            return False, (r.get("traceback") or r["error"])[-2500:]
        return False, "The script ran but produced no mesh geometry (0 triangles). It must create the 3D object."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--work", default="/content/t2a")
    p.add_argument("--blender", default="/content/blender/blender")
    p.add_argument("--tasks", default="", help="id phẩy; trống = tất cả trong tasks.jsonl")
    p.add_argument("--settings", default="text")
    p.add_argument("--max-rounds", type=int, default=3, help="số lượt sửa sau lượt đầu")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--exec-timeout", type=int, default=240)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--extra-body", default='{"reasoning_effort": "high"}')
    a = p.parse_args()

    tasks = [json.loads(l) for l in open(os.path.join(a.work, "tasks.jsonl"), encoding="utf-8")]
    if a.tasks:
        keep = set(a.tasks.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    sysp = {s: open(os.path.join(a.work, "bench", f"{s}_to_3d_system_prompt.txt"), encoding="utf-8").read()
            for s in ("text", "image")}
    os.makedirs(os.path.join(a.work, "gen"), exist_ok=True)
    out_final = os.path.join(a.work, "gen", f"{a.tag}.jsonl")
    out_r0 = os.path.join(a.work, "gen", f"{a.tag}_r0.jsonl")
    done = set()
    if os.path.exists(out_final):
        done = {(json.loads(l)["task"], json.loads(l)["setting"]) for l in open(out_final, encoding="utf-8")}
    url = a.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    extra = json.loads(a.extra_body) if a.extra_body else {}
    lock = threading.Lock()

    def chat(messages):
        t = time.time()
        r = requests.post(url, headers=headers, timeout=7200, json={
            "model": a.model, "messages": messages, "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 0,
            "max_tokens": a.max_tokens, **extra})
        r.raise_for_status()
        j = r.json()
        msg = j["choices"][0]["message"]
        reasoning, text = msg.get("reasoning_content") or "", msg.get("content") or ""
        ntok = j.get("usage", {}).get("completion_tokens", 0)
        return text, reasoning, ntok, j["choices"][0].get("finish_reason"), time.time() - t

    def one(t, setting):
        if setting == "image":
            content = [{"type": "image_url", "image_url": {"url": img_url(pth, a.img_size)}} for pth in t["images"]]
            content.append({"type": "text", "text": t["image_prompt"]})
        else:
            content = [{"type": "text", "text": t["text_prompt"]}]
        messages = [{"role": "system", "content": sysp[setting]}, {"role": "user", "content": content}]
        rounds, code, ok = [], "", False
        for rnd in range(a.max_rounds + 1):
            text, reasoning, ntok, fin, dt = chat(messages)
            raw = (f"<think>{reasoning}</think>" if reasoning else "") + text
            code = extract_code(raw)
            ok, err = try_exec(a.blender, code, a.exec_timeout) if code.strip() else (False, "No ```python code block found.")
            rounds.append({"round": rnd, "out_tokens": ntok, "seconds": round(dt, 1), "tok_s": round(ntok / max(dt, 1e-6), 1),
                           "finish_reason": fin, "ok": ok, "error": err[:600]})
            print(f"[{a.tag}] {t['id']} {setting} lượt {rnd}: {ntok} tok, {dt:.0f}s ({ntok / max(dt, 1e-6):.1f} tok/s), "
                  f"{'CHẠY ĐƯỢC' if ok else 'lỗi: ' + err.strip().splitlines()[-1][:120] if err.strip() else 'lỗi'}", flush=True)
            base = {"tag": a.tag, "model": a.model, "task": t["id"], "setting": setting, "thinking": "on"}
            if rnd == 0:
                with lock, open(out_r0, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**base, "tag": f"{a.tag}_r0", "raw": raw, "code": code, "out_tokens": ntok,
                                        "finish_reason": fin}, ensure_ascii=False) + "\n")
            if ok:
                break
            # giữ lịch sử ngắn: chỉ đưa code (không đưa phần thinking) + traceback
            messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                         {"role": "user", "content": FIX_PROMPT.format(err=err)}]
        rec = {**base, "raw": raw, "code": code, "out_tokens": sum(r["out_tokens"] for r in rounds),
               "finish_reason": rounds[-1]["finish_reason"], "rounds": rounds, "n_rounds": len(rounds), "fixed_ok": ok}
        with lock, open(out_final, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ok

    jobs = [(t, s) for s in a.settings.split(",") for t in tasks if (t["id"], s) not in done]
    nok = 0
    with ThreadPoolExecutor(a.parallel) as ex:
        futs = [ex.submit(one, t, s) for t, s in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                nok += bool(f.result())
            except Exception as e:
                print(f"[{a.tag}] lỗi: {type(e).__name__}: {str(e)[:200]}", flush=True)
            print(f"[gen] {k}/{len(futs)} xong, {nok} chạy được sau sửa", flush=True)


if __name__ == "__main__":
    main()
