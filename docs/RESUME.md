# Checkpoint & cách chạy tiếp (cập nhật 2026-11-27 ~07:00)

## Trạng thái Mốc 0

| Phần | Trạng thái |
|---|---|
| Dữ liệu, render GT, execset (3DCodeData trên Blender 5.0) | ✅ xong |
| Qwen3.5-4B, Qwen3.5-9B, Gemma 4 26B-A4B (zero-shot + chấm điểm) | ✅ xong → `results/phase0/results/summary.md` |
| Qwen3.8-27B NariQ 3-bit (llama.cpp fork) | ⏸ dừng ở 16/200 mẫu (đã sao lưu) — runtime đã trả lúc 06:57 |
| Gemma 4 12B, Gemma 4 31B-w4a16 | ⏸ chưa chạy |

Kết quả dở dang luôn được sao lưu về `results/phase0_sync/gen/*.jsonl` (mỗi ~10 phút);
khi chạy lại, các mẫu đã có được tự bỏ qua.

## Nếu máy vẫn bật trong lúc vắng mặt
`tools/chain_extra.sh` (chạy nền) tự: chờ 27B xong → chạy Gemma 12B/31B → tải kết quả về `results/phase0/` → **trả runtime**.
Xem tiến độ: `tail supervise.log`.

## Nếu máy đã tắt / cần chạy tiếp từ đầu phiên mới
```bash
cd D:/Documents/text2asset
# 1) Runtime cũ còn không? (còn thì tải kết quả rồi trả để khỏi tốn compute units)
python -I tools/watch_gpu_api.py --list            # kiểm tra API/đăng nhập
python -I tools/colab_job.py fetch --runtime text2asset --dest results/phase0_sync   # nếu runtime còn
python -I tools/watch_gpu_api.py --release --name text2asset

# 2) Chạy phần còn thiếu trên runtime mới (tự xin A100/G4, tự khôi phục gen đã có, tự trả runtime khi xong):
bash tools/supervise_phase0.sh >> supervise.log 2>&1        # MODELS mặc định gồm cả Gemma 12B/31B
#    27B NariQ cần script riêng (llama.cpp): sau khi có runtime ->
python -I tools/colab_job.py start --runtime text2asset --script run_27b_nariq.sh
#    (run_27b_nariq.sh tự tải llama.cpp fork + GGUF trên VM mới qua nari_setup.sh)
```
Hoặc chỉ cần mở Claude Code trong thư mục này (`claude --continue`) và nói "chạy tiếp Mốc 0".

## Hạ tầng (xem thêm memory colab-gpu-infra)
- Colab REST API qua ADC, quota project `text2asset-colab-701934`; đăng nhập lại nếu hết hạn:
  `gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory`
- Khởi động Claude Code với `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` để tác vụ nền không bị dừng khi thiếu RAM.

## Phát hiện chính tới giờ
- Gemma 4 26B-A4B zero-shot chạy được 62% (image) / 38% (text); Qwen3.5-4B/9B ≈ 0–4% (gọi sai API Blender 5.0,
  thinking quá dài bị cắt ở 32K). → Cân nhắc đổi student sang **Gemma 4 12B (dense)**, teacher **Gemma 4 31B** —
  chờ số liệu zero-shot của hai model này.
- 3DCodeData chạy được trên Blender 5.0: 89% (bản đầy đủ), 94% (`_geo`); GT benchmark 98%.
