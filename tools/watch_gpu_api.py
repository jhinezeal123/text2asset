"""Canh GPU trên Colab bằng Colab REST API (colaboratory.googleapis.com/v1beta) — chạy được trên Windows.

Xác thực bằng Application Default Credentials (gcloud), một lần:
    gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

Phụ thuộc: pip install google-auth requests

Ví dụ:
    python tools/watch_gpu_api.py --list                 # xem các cấu hình runtime và tài khoản có đủ điều kiện không
    python tools/watch_gpu_api.py                        # canh A100 mỗi 10 phút, tối đa 12 giờ
    python tools/watch_gpu_api.py --gpu A100 --high-mem --interval 5 --fallback L4
    python tools/watch_gpu_api.py --release              # trả runtime để ngừng tốn compute units

Mỗi kết quả in một dòng có marker để monitor bắt: STATUS=RETRY | STATUS=ACQUIRED | STATUS=TIMEOUT | STATUS=FATAL.
Khi xin được: ghi thông tin kết nối (URL + token proxy, là BÍ MẬT) vào .colab_runtime.json
(đã gitignore). Runtime tốn compute units cho tới khi --release.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
import uuid
from pathlib import Path

try:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
except ImportError:
    sys.exit("Thiếu thư viện: pip install google-auth requests")

API = "https://colaboratory.googleapis.com"
SCOPES = ["https://www.googleapis.com/auth/colaboratory"]
ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "watch_gpu_api.log"
CONN_FILE = ROOT / ".colab_runtime.json"


class Fatal(Exception):
    """Lỗi không nên thử lại (xác thực, không đủ điều kiện, sai tham số)."""


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def session(quota_project: str | None) -> AuthorizedSession:
    try:
        creds, _ = google.auth.default(scopes=SCOPES, quota_project_id=quota_project)
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise Fatal(f"Chưa có ADC. Chạy lệnh gcloud trong docstring trước.\n{e}")
    return AuthorizedSession(creds)


def call(s: AuthorizedSession, method: str, path: str, **kw) -> dict:
    r = s.request(method, API + path, timeout=120, **kw)
    try:
        body = r.json() if r.content else {}
    except ValueError:
        body = {"raw": r.text[:500]}
    if r.status_code in (401, 403):
        raise Fatal(f"HTTP {r.status_code} ở {path}: {json.dumps(body, ensure_ascii=False)[:800]}\n"
                    "→ Kiểm tra scope colaboratory trong ADC, hoặc tài khoản chưa được cấp quyền dùng Colab API.")
    if r.status_code >= 400:
        return {"_http_error": r.status_code, **body}
    return body


def list_specs(s: AuthorizedSession) -> list[dict]:
    body = call(s, "GET", "/v1beta/runtimespecs")
    if "_http_error" in body:
        raise Fatal(f"Không đọc được runtimespecs: {body}")
    return body.get("runtimeSpecs", [])


def pick_spec(specs: list[dict], gpu: str, high_mem: bool) -> dict:
    want_shape = "SHAPE_HIGHMEM" if high_mem else "SHAPE_STANDARD"
    matches = [sp for sp in specs
               if sp.get("key", {}).get("variant") == "VARIANT_GPU"
               and sp.get("key", {}).get("accelerator", "").upper() == gpu.upper()]
    if not matches:
        raise Fatal(f"Không có cấu hình GPU '{gpu}'. Dùng --list để xem các lựa chọn.")
    exact = [sp for sp in matches if sp["key"].get("shape") == want_shape]
    chosen = (exact or matches)[0]
    if not chosen.get("eligible"):
        raise Fatal(f"Tài khoản không đủ điều kiện dùng {chosen['key']} (cần gói Colab phù hợp/compute units).")
    return chosen["key"]


def get_runtime(s: AuthorizedSession, runtime_id: str) -> dict | None:
    body = call(s, "GET", f"/v1beta/runtimes/{runtime_id}")
    return None if "_http_error" in body else body


def wait_operation(s: AuthorizedSession, op: dict, timeout_s: int = 900) -> dict:
    deadline = time.time() + timeout_s
    while not op.get("done"):
        if time.time() > deadline:
            return {"done": True, "error": {"code": -1, "message": "operation timeout"}}
        time.sleep(10)
        op = call(s, "GET", f"/v1/{op['name']}")
        if "_http_error" in op:
            return {"done": True, "error": {"code": op["_http_error"], "message": str(op)[:500]}}
    return op


def try_create(s: AuthorizedSession, runtime_id: str, key: dict) -> tuple[dict | None, str]:
    op = call(s, "POST", "/v1beta/runtimes",
              params={"runtimeId": runtime_id, "requestId": str(uuid.uuid4())},
              json={"runtimeSpec": key})
    if "_http_error" in op:
        err = op.get("error", op)
        if op["_http_error"] == 400:
            raise Fatal(f"Request sai (400): {json.dumps(err, ensure_ascii=False)[:800]}")
        return None, f"HTTP {op['_http_error']}: {json.dumps(err, ensure_ascii=False)[:500]}"
    op = wait_operation(s, op)
    if "error" in op:
        e = op["error"]
        if e.get("code") == 6:  # ALREADY_EXISTS -> dùng runtime đang có nếu đúng GPU
            rt = get_runtime(s, runtime_id)
            if rt and rt.get("runtimeSpec", {}).get("accelerator", "").upper() == key["accelerator"].upper():
                return rt, "đã tồn tại"
            return None, f"runtime '{runtime_id}' đã tồn tại với cấu hình khác: {rt and rt.get('runtimeSpec')}"
        reasons = [d.get("reason") for d in e.get("details", []) if isinstance(d, dict) and d.get("reason")]
        return None, f"code {e.get('code')}: {e.get('message', '')[:300]} {reasons or ''}"
    rt = op.get("response") or get_runtime(s, runtime_id)
    return rt, "ok"


def save_conn(rt: dict) -> None:
    CONN_FILE.write_text(json.dumps(rt, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_candidates(gpu_arg: str, default_hm: bool) -> list[tuple[str, bool]]:
    """'A100:hm,A100,G4' -> [('A100', True), ('A100', False), ('G4', False)]; ':std' ép bản thường."""
    out = []
    for item in gpu_arg.split(","):
        name, _, shape = item.strip().partition(":")
        hm = default_hm if not shape else shape.lower() in ("hm", "highmem", "high-mem")
        out.append((name.upper(), hm))
    return out


def acquire(s, a, gpu: str, deadline: float) -> dict | None:
    specs = list_specs(s)
    keys = []
    for name, hm in parse_candidates(gpu, a.high_mem):
        try:
            k = pick_spec(specs, name, hm)
        except Fatal as e:
            log(f"Bỏ qua {name}{':hm' if hm else ''}: {e}")
            continue
        if k not in keys:
            keys.append(k)
    if not keys:
        raise Fatal(f"Không có cấu hình nào dùng được trong '{gpu}'.")
    log("Thứ tự thử mỗi lượt: " + " → ".join(f"{k['accelerator']}/{k['shape'].replace('SHAPE_', '')}" for k in keys))
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        msgs = []
        for key in keys:
            label = f"{key['accelerator']}/{key['shape'].replace('SHAPE_', '')}"
            log(f"Lần {attempt}: xin {label} ...")
            try:
                rt, msg = try_create(s, a.name, key)
            except Fatal:
                raise
            except Exception as e:  # lỗi mạng tạm thời -> thử lại ở lần sau
                rt, msg = None, f"lỗi tạm thời {type(e).__name__}: {e}"
            if rt:
                return rt
            msgs.append(f"{label}: {msg}")
        msg = " || ".join(msgs)
        log(f"STATUS=RETRY lần {attempt} chưa được: {msg}")
        wait = a.interval * 60 * random.uniform(0.85, 1.15)
        log(f"Thử lại sau {wait / 60:.1f} phút.")
        time.sleep(min(wait, max(0, deadline - time.time())))
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="A100",
                   help="một hoặc nhiều GPU theo thứ tự ưu tiên, vd 'A100:hm,A100,G4' (:hm = High-RAM)")
    p.add_argument("--name", default="text2asset", help="runtimeId (chữ thường, số, gạch ngang)")
    p.add_argument("--interval", type=float, default=10, help="phút giữa các lần thử")
    p.add_argument("--max-hours", type=float, default=12)
    p.add_argument("--high-mem", action="store_true")
    p.add_argument("--fallback", default=None, help="GPU dự phòng khi hết giờ, vd L4")
    p.add_argument("--quota-project", default=None, help="GCP project tính quota nếu API đòi")
    p.add_argument("--list", action="store_true", help="liệt kê runtimespecs rồi thoát")
    p.add_argument("--release", action="store_true", help="xóa runtime --name rồi thoát")
    a = p.parse_args()

    try:
        s = session(a.quota_project)
        if a.list:
            for sp in list_specs(s):
                k = sp.get("key", {})
                print(f"{'OK ' if sp.get('eligible') else ' - '} {k.get('variant'):<12} "
                      f"{k.get('accelerator'):<6} {k.get('shape')}")
            return 0
        if a.release:
            body = call(s, "DELETE", f"/v1beta/runtimes/{a.name}", params={"allowMissing": "true"})
            log(f"Đã trả runtime '{a.name}'." if "_http_error" not in body else f"Lỗi khi xóa: {body}")
            CONN_FILE.unlink(missing_ok=True)
            return 0

        existing = get_runtime(s, a.name)
        if existing:
            log(f"Runtime '{a.name}' đang tồn tại: {existing.get('runtimeSpec')}")
            wanted = {n for n, _ in parse_candidates(a.gpu, a.high_mem)}
            if existing.get("runtimeSpec", {}).get("accelerator", "").upper() in wanted:
                save_conn(existing)
                log(f"STATUS=ACQUIRED runtime {existing.get('name')} — {existing.get('runtimeSpec')} (đã có sẵn).")
                return 0
            log("STATUS=FATAL runtime cùng tên đang dùng GPU khác → chạy --release trước nếu muốn xin lại.")
            return 1

        deadline = time.time() + a.max_hours * 3600
        rt = acquire(s, a, a.gpu, deadline)
        if not rt and a.fallback:
            log(f"Hết {a.max_hours} giờ chưa có {a.gpu}, thử {a.fallback} ...")
            rt = acquire(s, a, a.fallback, time.time() + 1)  # đúng một lần thử
        if not rt:
            log(f"STATUS=TIMEOUT hết {a.max_hours} giờ mà chưa xin được {a.gpu}.")
            return 1

        save_conn(rt)
        spec = rt.get("runtimeSpec", {})
        log(f"STATUS=ACQUIRED runtime {rt.get('name')} — {spec}. Thông tin kết nối lưu ở {CONN_FILE.name}.")
        return 0
    except Fatal as e:
        log(f"STATUS=FATAL {e}")
        return 2
    except Exception as e:  # lỗi bất ngờ (mạng, parse...) cũng phải báo cho monitor
        log(f"STATUS=FATAL lỗi bất ngờ: {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
