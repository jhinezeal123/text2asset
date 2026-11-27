"""Điều khiển job trên runtime Colab (tạo qua tools/watch_gpu_api.py) bằng Jupyter API qua proxy Colab.

python tools/colab_job.py start   [--runtime text2asset] [--env SMOKE=1]   # tải code colab/phase0 lên & chạy nền
python tools/colab_job.py monitor [--interval 120]   # in dòng STAGE/WARN/... mới; thoát khi PHASE0_DONE/FAILED
python tools/colab_job.py status                     # in 40 dòng log cuối
python tools/colab_job.py fetch   [--dest results/phase0]   # tải results/ + gen/ + log về máy
python tools/colab_job.py exec "print(1)"            # chạy nhanh một đoạn code

Thiết kế (do proxy Colab hay cắt websocket sau vài message):
- Gửi lệnh qua websocket, chỉ cần kernel xác nhận đã nhận (execute_input/busy) là xong.
- Kết quả/log đọc qua Jupyter Contents API (HTTP; gốc là "/" của VM).
Job chạy trong một kernel riêng (kernel bận -> runtime không bị thu hồi vì idle). Log: /content/t2a/run.log
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import re
import sys
import tarfile
import time
import uuid
from pathlib import Path

import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watch_gpu_api as wg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".colab_job.json"
SYNC_DIR = ROOT / "results" / "phase0_sync"  # bản sao kết quả dở dang (monitor tự cập nhật)
REMOTE = "/content/t2a"
LOG = f"{REMOTE}/run.log"
EVENT_RE = re.compile(r"\] STAGE |\] WARN |PHASE0_|MODEL_FAILED|Traceback|CUDA out of memory|\[gen\] \d+/\d+ |"
                      r"\[execset\] \d+/|mẫu, |nạp model")


class Runtime:
    def __init__(self, runtime_id: str):
        self.id = runtime_id
        self.s = wg.session(None)
        self.refresh()

    # ------------------------------------------------------------ kết nối
    def refresh(self):
        rt = wg.get_runtime(self.s, self.id)
        if not rt:
            raise SystemExit(f"RUNTIME_LOST runtime '{self.id}' không tồn tại")
        ci = rt["connectionInfo"]
        self.base = ci["url"].rstrip("/")
        self.token = ci["token"]
        # Giống Colab CLI chính thức: token trong header + query param, kèm client agent.
        self.hdr = {"X-Colab-Runtime-Proxy-Token": ci["token"], "X-Colab-Client-Agent": "colab-cli"}
        self.expire = dt.datetime.fromisoformat(ci["expireTime"].replace("Z", "+00:00"))

    def ensure_fresh(self):
        if self.expire - dt.datetime.now(dt.timezone.utc) < dt.timedelta(minutes=10):
            self.refresh()

    def req(self, method, path, params=None, **kw):
        self.ensure_fresh()
        p = {"colab-runtime-proxy-token": self.token, **(params or {})}
        return self.s.request(method, f"{self.base}{path}", headers=self.hdr, params=p, timeout=120, **kw)

    def http(self, method, path, **kw):
        r = self.req(method, path, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    # ------------------------------------------------------------ file (Contents API)
    def read_file(self, path: str, binary: bool = False) -> bytes | str | None:
        r = self.req("GET", f"/api/contents{path}",
                     params={"content": 1, "type": "file", "format": "base64" if binary else "text"})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        c = r.json()["content"]
        return base64.b64decode(c) if binary else c

    def delete_file(self, path: str):
        try:
            self.req("DELETE", f"/api/contents{path}")
        except Exception:
            pass

    # ------------------------------------------------------------ kernel
    def kernels(self) -> list[dict]:
        return self.http("GET", "/api/kernels")

    def new_kernel(self) -> str:
        # Không chờ "idle": Jupyter giữ kernel ở "starting" cho tới khi có websocket kết nối.
        return self.http("POST", "/api/kernels", json={"name": "python3"})["id"]

    def connect(self, kid: str, timeout: int = 300):
        """Mở websocket tới kernel; thử lại khi tiến trình kernel chưa lên (404/502)."""
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            self.ensure_fresh()
            url = (self.base.replace("https://", "wss://") + f"/api/kernels/{kid}/channels"
                   f"?session_id={uuid.uuid4().hex}&colab-runtime-proxy-token={self.token}")
            try:
                return websocket.create_connection(url, header=[f"{k}: {v}" for k, v in self.hdr.items()], timeout=60)
            except (websocket.WebSocketBadStatusException, websocket.WebSocketConnectionClosedException, OSError) as e:
                last = e
                time.sleep(5)
        raise TimeoutError(f"không kết nối được kernel {kid}: {last}")

    def helper_kernel(self) -> str:
        """Kernel phụ để chạy lệnh ngắn (khác kernel đang chạy job)."""
        job_kid = json.loads(STATE.read_text()).get("kernel") if STATE.exists() else None
        for k in self.kernels():
            if k["id"] != job_kid and k.get("execution_state") == "idle":
                return k["id"]
        return self.new_kernel()

    def send(self, kid: str, code: str, timeout: int = 180):
        """Gửi execute_request; trả về khi kernel xác nhận đã nhận lệnh."""
        ws = self.connect(kid)
        mid = uuid.uuid4().hex
        try:
            ws.send(json.dumps({
                "header": {"msg_id": mid, "username": "t2a", "session": uuid.uuid4().hex,
                           "msg_type": "execute_request", "version": "5.3",
                           "date": dt.datetime.now(dt.timezone.utc).isoformat()},
                "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
                "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {},
                            "allow_stdin": False, "stop_on_error": True}}))
            t0 = time.time()
            while time.time() - t0 < timeout:
                m = json.loads(ws.recv())
                if m.get("parent_header", {}).get("msg_id") == mid and (
                        m["msg_type"] == "execute_input" or
                        (m["msg_type"] == "status" and m["content"]["execution_state"] == "busy")):
                    return
            raise TimeoutError("kernel không xác nhận lệnh")
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def run(self, code: str, timeout: int = 900) -> str:
        """Chạy code trên kernel phụ, lấy stdout qua file (không phụ thuộc websocket)."""
        rid = uuid.uuid4().hex[:12]
        out, done = f"/tmp/t2a_{rid}.out", f"/tmp/t2a_{rid}.done"
        wrapped = (
            "import contextlib as _c, io as _io, traceback as _t\n"
            "_b = _io.StringIO()\n"
            "try:\n"
            "    with _c.redirect_stdout(_b), _c.redirect_stderr(_b):\n"
            f"        exec({code!r}, {{'__name__': '__t2a__'}})\n"
            "except BaseException:\n"
            "    _b.write(_t.format_exc())\n"
            f"open({out!r}, 'w').write(_b.getvalue())\n"
            f"open({done!r}, 'w').write('1')\n")
        self.send(self.helper_kernel(), wrapped)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.read_file(done) is not None:
                text = self.read_file(out) or ""
                self.delete_file(out)
                self.delete_file(done)
                return text
            time.sleep(2)
        raise TimeoutError(f"lệnh chưa xong sau {timeout}s")


def pack_code() -> str:
    """Đóng gói colab/phase0, ép xuống dòng LF (file CRLF làm bash trên VM chết ngay)."""
    buf = io.BytesIO()
    src = ROOT / "colab" / "phase0"
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(src.iterdir()):
            if not f.is_file() or f.suffix not in (".py", ".sh"):
                continue
            data = f.read_bytes().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(f"phase0/{f.name}")
            ti.size, ti.mode, ti.mtime = len(data), 0o755, int(time.time())
            tf.addfile(ti, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def cmd_start(rt: Runtime, a):
    env = dict(kv.split("=", 1) for kv in a.env)
    print(rt.run(f"""
import base64, io, os, tarfile
os.makedirs({REMOTE!r}, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode({pack_code()!r}))).extractall({REMOTE!r} + '/code')
print('uploaded:', sorted(os.listdir({REMOTE!r} + '/code/phase0')))
""", timeout=600).strip())
    gen_dir = SYNC_DIR / "gen"
    if gen_dir.exists() and any(gen_dir.glob("*.jsonl")):
        # Khôi phục kết quả sinh code từ lần chạy trước -> zeroshot_gen tự bỏ qua mục đã có.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(gen_dir, arcname="gen")
        print(rt.run(f"""
import base64, io, os, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buf.getvalue()).decode()!r}))).extractall({REMOTE!r})
print('restored gen:', sorted(os.listdir({REMOTE!r} + '/gen')))
""", timeout=600).strip())
    kid = rt.new_kernel()
    STATE.write_text(json.dumps({"runtime": rt.id, "kernel": kid, "started": time.time()}))
    rt.send(kid, f"""
import os, subprocess
env = dict(os.environ, **{env!r})
rc = subprocess.run('bash {REMOTE}/code/phase0/{a.script} >> {LOG} 2>&1', shell=True, env=env).returncode
log = open({LOG!r}, encoding='utf-8', errors='replace').read()
if rc != 0 and 'PHASE0_FAILED' not in log and 'PHASE0_DONE' not in log[-2000:]:
    with open({LOG!r}, 'a') as f:
        f.write(f'PHASE0_FAILED script thoát bất thường rc={{rc}}\\n')
""")
    print(f"JOB_STARTED kernel={kid} log={LOG}")


def cmd_monitor(rt: Runtime, a):
    seen, errors = None if a.from_end else 0, 0
    last_sync = 0.0
    while True:
        try:
            if time.time() - last_sync > a.sync_every:
                # Sao lưu kết quả dở dang về máy: runtime Colab có thể bị thu hồi bất cứ lúc nào.
                try:
                    fetch_rels(rt, ["gen", "results", "run.log", "execset.log"], SYNC_DIR)
                    last_sync = time.time()
                except Exception as e:
                    print(f"MONITOR_ERR sync: {type(e).__name__}: {str(e)[:150]}", flush=True)
            text = rt.read_file(LOG) or ""
            errors = 0
            lines = text.splitlines()
            if seen is None:  # --from-end: bỏ qua log cũ
                seen = len(lines)
            for line in lines[seen:]:
                if EVENT_RE.search(line):
                    print(line[:300], flush=True)
                if "PHASE0_DONE" in line:
                    return 0
                if "PHASE0_FAILED" in line:
                    return 1
            seen = len(lines)
        except SystemExit as e:  # runtime không còn
            print(str(e), flush=True)
            return 4
        except Exception as e:
            errors += 1
            if errors in (1, 5) or errors % 20 == 0:
                print(f"MONITOR_ERR lần {errors}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            try:
                rt.refresh()
            except SystemExit as e2:
                print(str(e2), flush=True)
                return 4
            except Exception:
                pass
        time.sleep(a.interval)


def cmd_status(rt: Runtime, a):
    text = rt.read_file(LOG)
    print("\n".join((text or "(chưa có log)").splitlines()[-40:]))


def fetch_rels(rt: Runtime, rels: list[str], dest: Path) -> int:
    tgz = f"/tmp/t2a_fetch_{uuid.uuid4().hex[:8]}.tar.gz"
    rt.run(f"""
import os, tarfile
with tarfile.open({tgz!r}, 'w:gz') as tf:
    for rel in {rels!r}:
        p = os.path.join({REMOTE!r}, rel)
        if os.path.exists(p): tf.add(p, arcname=rel)
""", timeout=900)
    data = rt.read_file(tgz, binary=True) or b""
    rt.delete_file(tgz)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")
    return len(data)


def cmd_fetch(rt: Runtime, a):
    rels = ["results", "gen", "run.log", "execset.log", "tasks.jsonl"] + (["render"] if a.with_render else [])
    n = fetch_rels(rt, rels, ROOT / a.dest)
    print(f"FETCHED {n / 1e6:.1f} MB -> {ROOT / a.dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["start", "monitor", "status", "fetch", "exec"])
    p.add_argument("code", nargs="?")
    p.add_argument("--runtime", default="text2asset")
    p.add_argument("--env", nargs="*", default=[])
    p.add_argument("--interval", type=float, default=120)
    p.add_argument("--from-end", action="store_true", help="monitor: chỉ in dòng mới từ bây giờ")
    p.add_argument("--sync-every", type=float, default=600, help="monitor: giây giữa các lần sao lưu kết quả về máy")
    p.add_argument("--dest", default="results/phase0")
    p.add_argument("--with-render", action="store_true", help="fetch: tải cả ảnh render + points (render/)")
    p.add_argument("--script", default="run_phase0.sh", help="start: script trong colab/phase0 để chạy")
    a = p.parse_args()
    rt = Runtime(a.runtime)
    if a.cmd == "start":
        cmd_start(rt, a)
    elif a.cmd == "monitor":
        sys.exit(cmd_monitor(rt, a))
    elif a.cmd == "status":
        cmd_status(rt, a)
    elif a.cmd == "fetch":
        cmd_fetch(rt, a)
    elif a.cmd == "exec":
        print(rt.run(a.code))


if __name__ == "__main__":
    main()
