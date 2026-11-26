"""Canh A100 trên Colab: thử xin runtime định kỳ bằng Colab CLI cho tới khi được.

Yêu cầu: đã cài và đăng nhập Colab CLI (`colab`), xem hướng dẫn trong README của repo
https://github.com/googlecolab/google-colab-cli .

Ví dụ:
    python tools/watch_a100.py                      # thử mỗi 10 phút, tối đa 12 giờ
    python tools/watch_a100.py --gpu A100 --interval 5 --max-hours 6 --high-mem
    python tools/watch_a100.py --fallback L4        # hết hạn mà chưa có A100 thì lấy L4

Khi xin được: in thông tin runtime, mở runtime trong trình duyệt (`colab url --open`),
kêu bíp + hiện hộp thoại (Windows). LƯU Ý: runtime đã cấp sẽ tốn compute units cho tới khi
bạn dùng hoặc `colab stop -s <tên>`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import shutil
import subprocess
import sys
import time

LOG_FILE = "watch_a100.log"


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def session_has_gpu(colab: str, name: str, gpu: str) -> tuple[bool, str]:
    """Kiểm tra session `name` đang tồn tại và đúng loại GPU (đọc output `colab status`)."""
    try:
        r = run([colab, "status", "-s", name], timeout=120)
    except subprocess.TimeoutExpired:
        return False, "status timeout"
    out = (r.stdout + r.stderr).strip()
    return (r.returncode == 0 and gpu.lower() in out.lower()), out


def try_allocate(colab: str, name: str, gpu: str, high_mem: bool) -> tuple[bool, str]:
    cmd = [colab, "new", "-s", name, "--gpu", gpu]
    if high_mem:
        cmd.append("--high-mem")
    try:
        r = run(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "colab new timeout (600s)"
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return False, out
    # Exit code cho lỗi thiếu GPU không được tài liệu hóa -> xác nhận lại bằng status.
    ok, status = session_has_gpu(colab, name, gpu)
    return ok, out + "\n--- status ---\n" + status


def notify(title: str, text: str) -> None:
    if sys.platform == "win32":
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 400)
                time.sleep(0.2)
        except Exception:
            pass
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x40 | 0x1000)  # info + topmost
        except Exception:
            pass
    else:
        print("\a", end="", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="A100", help="T4, L4, G4, H100, A100 (mặc định A100)")
    p.add_argument("--name", default="text2asset", help="tên session Colab CLI")
    p.add_argument("--interval", type=float, default=10, help="phút giữa các lần thử (mặc định 10)")
    p.add_argument("--max-hours", type=float, default=12, help="dừng sau bao nhiêu giờ (mặc định 12)")
    p.add_argument("--high-mem", action="store_true", help="xin High-RAM (cần Pro/Pro+)")
    p.add_argument("--fallback", default=None, help="GPU dự phòng khi hết giờ mà chưa có (vd L4)")
    p.add_argument("--no-open", action="store_true", help="không tự mở runtime trong trình duyệt")
    a = p.parse_args()

    colab = shutil.which("colab")
    if not colab:
        log("Không tìm thấy lệnh `colab`. Cài Colab CLI và đăng nhập trước.")
        return 2

    ok, status = session_has_gpu(colab, a.name, a.gpu)
    if ok:
        log(f"Session '{a.name}' đã có {a.gpu} sẵn:\n{status}")
        return 0

    try:
        usage = run([colab, "usage"], timeout=120)
        log("Compute units:\n" + (usage.stdout + usage.stderr).strip())
    except Exception as e:  # không chặn việc canh nếu lệnh này lỗi
        log(f"Không đọc được `colab usage`: {e}")

    deadline = time.time() + a.max_hours * 3600
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        log(f"Lần {attempt}: xin {a.gpu} ...")
        ok, out = try_allocate(colab, a.name, a.gpu, a.high_mem)
        if ok:
            log(f"ĐÃ CÓ {a.gpu} (session '{a.name}'):\n{out}")
            if not a.no_open:
                run([colab, "url", "-s", a.name, "--open"], timeout=120)
            notify("Colab", f"Đã xin được {a.gpu}! Session: {a.name}\n"
                            f"Nhớ `colab stop -s {a.name}` khi không dùng để khỏi tốn compute units.")
            return 0
        last = out.splitlines()[-3:] if out else ["(không có output)"]
        log("Chưa được: " + " | ".join(last))
        # Phòng trường hợp `new` thất bại nhưng vẫn để lại session CPU/sai GPU -> dọn để lần sau xin lại.
        run([colab, "stop", "-s", a.name], timeout=120)
        sleep_s = a.interval * 60 * random.uniform(0.85, 1.15)
        log(f"Thử lại sau {sleep_s / 60:.1f} phút.")
        time.sleep(sleep_s)

    if a.fallback:
        log(f"Hết {a.max_hours} giờ chưa có {a.gpu}, thử GPU dự phòng {a.fallback} ...")
        ok, out = try_allocate(colab, a.name, a.fallback, a.high_mem)
        if ok:
            log(f"ĐÃ CÓ {a.fallback}:\n{out}")
            if not a.no_open:
                run([colab, "url", "-s", a.name, "--open"], timeout=120)
            notify("Colab", f"Không có {a.gpu}, đã lấy {a.fallback}. Session: {a.name}")
            return 0
        log(f"Cũng không xin được {a.fallback}:\n{out}")

    log("Dừng canh: hết thời gian.")
    notify("Colab", f"Hết {a.max_hours} giờ mà chưa xin được {a.gpu}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
