# Kế Hoạch v2: Huấn Luyện VLM Sinh Code Blender Cho Roblox Studio

> Bản sửa từ `ke_hoach_train_3d_coder_roblox.md.docx` (v1). Thay đổi chính so với v1:
> bỏ backbone MoE (Gemma 4 26B-A4B) → **student Qwen3.5-9B (dense) + teacher Qwen3.8-27B**;
> thay GRPO online bằng **rejection sampling lặp** (GRPO ngắn là tùy chọn);
> thêm Mốc 0 chọn model bằng số liệu, dedup chống rò rỉ benchmark, sandbox cách ly thật.

**Mục tiêu:** Mô hình nhận ảnh tham chiếu đa góc và/hoặc mô tả văn bản, sinh Blender Python script (bpy) tạo 3D game asset xuất `.glb` / `.fbx` dùng được ngay trong Roblox Studio.

## Tổng quan cấu hình

| Hạng mục | Lựa chọn | Ghi chú |
|---|---|---|
| Student (model được train) | **Qwen3.5-9B** (dense, đa phương thức gốc, Apache 2.0); dự phòng Qwen3.5-4B nếu cần rollout nhanh hơn | LoRA bf16, không lượng tử hóa; cần transformers ≥ 5.2; chốt ở Mốc 0 |
| Teacher (sinh dữ liệu) | **Qwen3.8-27B** (dense VLM) | Chỉ dùng inference qua vLLM; kiểm tra license trên model card chính thức |
| Phần cứng | 1× A100 **80GB** (khuyến nghị) | 40GB vẫn chạy được student 8B nhưng chật khi sinh dài |
| Blender | 5.0 headless — **hoặc 4.x LTS** nếu tỷ lệ chạy được của dữ liệu cũ trên 5.0 quá thấp (chốt ở Mốc 0) | Roblox không phụ thuộc phiên bản Blender |
| Dữ liệu SFT | 3DCodeData + nhánh Blender 3DCodeFactory + dữ liệu teacher đã lọc | Dedup với 3DCodeBench |
| Dữ liệu RL/RFT | 2K–5K prop hard-surface CC-BY/CC0 (TRELLIS-500K, Cap3D, TexVerse) | Có mesh GT, không cần code GT |
| Đánh giá | 3DCodeBench (212 tác vụ) + 50 prop Roblox thực tế | Tách tập held-out cố định từ đầu |

### Chuẩn render thống nhất (dùng chung cho SFT, RL, eval)

- 4 góc phương vị 45°, 135°, 225°, 315°; **elevation cố định 25°**; camera cùng FOV và khoảng cách.
- Mesh chuẩn hóa về hộp đơn vị [-1, 1]³ trước khi render; nền trắng; ánh sáng cố định; 384×384.
- Ảnh GT và ảnh render từ code sinh ra phải đi qua **cùng một renderer**.

### Cấu trúc thư mục

```
text2asset/
├── data/
│   ├── sft_raw/            # 3DCodeData + 3DCodeFactory (Blender)
│   ├── teacher_gen/        # Mẫu do Qwen3.8-27B sinh, đã chấm reward
│   ├── sft_clean.jsonl
│   ├── rl_props/           # 2K–5K prop + ảnh render chuẩn
│   └── heldout/            # Tập validation cố định (không bao giờ train)
├── sandbox/
│   ├── Dockerfile          # Blender headless, không mạng
│   ├── blender_worker.py   # Pool worker, reset scene, timeout cứng
│   ├── renderer.py         # Render 4 góc theo chuẩn ở trên
│   ├── reward.py           # Reward phân tầng + luật Roblox
│   └── roblox_export.py
├── train/
│   ├── sft_lora.py
│   ├── rft_round.py        # Một vòng rejection sampling
│   └── grpo_rl.py          # Tùy chọn
└── eval/
    ├── zeroshot_compare.py # Mốc 0
    └── eval_3dcodebench.py
```

## Phát hiện từ dữ liệu (2026-11-26)

- 3DCodeData/3DCodeBench **viết sẵn cho Blender 5.0** (chưng cất từ Infinigen, BSD-3) → chốt Blender 5.0, bỏ so sánh 4.x.
- **Benchmark và tập train dùng chung 212 loại** (3DCodeData = 212 factory × 60 seed; benchmark = seed 0 của cùng 212 factory).
  Dedup theo mẫu là không đủ → cần **giữ lại hẳn một nhóm loại (held-out categories)** không train, và báo cáo điểm riêng trên nhóm đó.
- Code GT rất dài: mẫu 200 script (tokenizer Qwen3.5), bản đầy đủ p50 ≈ 10.6K token, chỉ ~43% ≤ 8K;
  bản `_geo` p50 ≈ 4.2K, ~70% ≤ 8K, ~96% ≤ 16K → **SFT ưu tiên bản `_geo`** và nâng max seq lên ~16K.
- Harness gốc: system prompt riêng cho text/image, temperature 0.7, max 16K token đầu ra, 240 s/script — Mốc 0 dùng y nguyên.

Pipeline Mốc 0 đã viết & smoke test (oracle đạt IoU/DINO 0.999): `colab/phase0/`, điều khiển bằng `tools/colab_job.py`,
tự khởi chạy khi xin được A100 bằng `tools/auto_phase0.sh`.

## Mốc 0 (Ngày 1–2): Chọn model & kiểm tra giả định bằng số liệu

- [ ] Khóa phiên bản môi trường: transformers, peft, trl, vllm, trimesh, open3d, Blender.
- [ ] **Đo tỷ lệ chạy được** của 3DCodeData / 3DCodeFactory trên Blender 5.0 (mẫu ngẫu nhiên 500 script). Nếu < ~60% → cân nhắc Blender 4.x LTS hoặc viết lớp sửa API tự động.
- [ ] **So sánh zero-shot** 2–3 ứng viên trên ~50 tác vụ 3DCodeBench (cả image→3D và text→3D):
  Qwen3.5-9B, Qwen3.5-4B, Qwen3.8-27B (teacher), (tham chiếu) Gemma 4 26B-A4B.
  Đo: executability, mask IoU / DINO similarity với ảnh ref, độ dài code trung bình, thời gian sinh — **chạy cả hai chế độ thinking bật/tắt**.
- [ ] Xác nhận stack hỗ trợ đủ dòng Qwen3.5/3.8 (kiến trúc attention lai): transformers ≥ 5.2, vLLM, PEFT/TRL target được các lớp linear-attention, có kernel tăng tốc (flash-linear-attention, causal-conv1d) — nếu thiếu kernel, training sẽ chậm đáng kể.
- [ ] Đọc license chính thức của Qwen3.8-27B (các nguồn đang mâu thuẫn); dự phòng teacher Qwen3.6-27B.
- [ ] **Go/No-Go:** chốt student, teacher, phiên bản Blender; ghi baseline zero-shot của student làm mốc so sánh.

## Mốc 1 (Ngày 2–5): Dữ liệu SFT sạch

- [ ] Chạy kiểm thử thực thi toàn bộ dữ liệu trong sandbox (timeout 20s/script): loại lỗi cú pháp/API, mesh rỗng (`n_meshes == 0` hoặc `tris == 0`), mẫu quá dài (> ~6K token code).
- [ ] **Render 4 ảnh chuẩn cho mọi mẫu SFT** bằng chính code của mẫu đó.
- [ ] **Dedup chống rò rỉ benchmark:** loại mẫu trùng/gần trùng với 212 tác vụ 3DCodeBench và tập `heldout/` (so prompt, hash ảnh, embedding ảnh, Chamfer mesh).
- [ ] **Sinh dữ liệu teacher:** Qwen3.8-27B sinh 4–8 mẫu/prop trên một phần tập prop game (khác tập RL/held-out), chấm bằng `reward.py`, chỉ giữ mẫu chạy được và vượt ngưỡng reward. Mục tiêu +2K–5K mẫu đúng domain game.
- [ ] Đóng gói `sft_clean.jsonl` theo chat template của student: ảnh trước text, drop ảnh ngẫu nhiên 25% (học cả text→3D), tắt chế độ thinking.
- [ ] **Go/No-Go:** 100% mẫu chạy ra mesh hợp lệ; 10K–20K mẫu; đã dedup.

## Mốc 2 (Ngày 2–7, chạy song song Mốc 1 & 3): Sandbox & reward

Phần này chủ yếu dùng CPU và công lập trình → làm song song khi GPU đang bận.

- [ ] **Cách ly thật:** worker chạy trong container, tắt mạng, chỉ mount thư mục tạm, giới hạn RAM/CPU; tiền kiểm chặn `import os/subprocess/socket/shutil` và đọc/ghi file ngoài thư mục tạm.
- [ ] `blender_worker.py`: pool 8–16 tiến trình, reset bằng `read_factory_settings(use_empty=True)`, timeout cứng 10–15s, tự khởi động lại khi treo/crash.
- [ ] `renderer.py`: Workbench/EEVEE hoặc rasterizer GPU (nvdiffrast/pyrender) theo đúng chuẩn render.
- [ ] `reward.py`:
  - **R_exec (cổng):** lỗi/timeout/mesh rỗng → tổng = 0.
  - **R_mask:** IoU silhouette trung bình 4 góc (rẻ, khó hack).
  - **R_vis:** cosine DINO (ưu tiên) / SigLIP-2 giữa render và ảnh ref.
  - **R_geo:** Chamfer sau chuẩn hóa, lấy tốt nhất trong 4 góc xoay trục Z.
  - **R_roblox (phạt):** > 5K tris (prop nhỏ) / > 15K tris (asset lớn), non-manifold nặng.
  - Khởi đầu: `R = R_exec · (0.3·R_mask + 0.4·R_vis + 0.3·R_geo) − R_roblox`; tinh chỉnh sau khi xem phân phối.
- [ ] Benchmark trên 100 script thật: báo **p50 / p95** thời gian chấm (không chỉ trung bình).
- [ ] Lọc tập prop RL (license CC-BY/CC0 từ metadata, hard-surface, loại nhân vật hữu cơ/scan) và render ảnh chuẩn.

## Mốc 3 (Ngày 6–9): SFT student

- [ ] LoRA bf16 trên student: r=32–64, alpha=2r, dropout 0.05, target toàn bộ linear của LLM; **đóng băng vision encoder**.
- [ ] LR 1e-4 (cosine, warmup 3–5%), 1–2 epoch, gradient checkpointing, max seq ~8K; đặt `max_pixels` để mỗi ảnh ~150–300 token.
- [ ] Dry-run 100 mẫu: peak VRAM, tốc độ step, loss giảm ổn định.
- [ ] Mỗi 200 step: lưới ảnh 4 góc của 16–20 mẫu held-out cố định.
- [ ] Chấm checkpoint trên 3DCodeBench (cả hai nhánh), so với baseline zero-shot ở Mốc 0.
- [ ] **Go/No-Go:** executability ≥ 80–85% và hình khối bám ảnh ref; nếu chưa đạt → quay lại dữ liệu.

## Mốc 4 (Ngày 10–16): Rejection sampling lặp (RFT)

Mỗi vòng tách rời sinh và train → không cần nạp vLLM và trainer cùng lúc trên 1 GPU.

- [ ] **Một vòng:**
  1. vLLM sinh N = 8 mẫu/prompt (temperature 0.8–1.0, giới hạn 2–4K token) trên tập prop RL.
  2. Chấm toàn bộ bằng pool Blender.
  3. Giữ mẫu tốt nhất mỗi prompt nếu vượt ngưỡng tuyệt đối và tốt hơn vòng trước.
  4. SFT tiếp từ checkpoint hiện tại trên dữ liệu mới + ~30% dữ liệu SFT gốc (chống quên).
  5. Đánh giá trên held-out; chỉ nhận checkpoint nếu cải thiện.
- [ ] Chạy thử 1 vòng trên 500–1.000 prop, sau đó 2–3 vòng trên toàn tập.
- [ ] Theo dõi từng thành phần reward riêng; xem lưới ảnh để phát hiện reward hacking (khối hộp/cầu tô màu na ná).
- [ ] *(Tùy chọn, nếu còn thời gian/GPU)* GRPO ngắn vài trăm step từ checkpoint RFT tốt nhất, G = 4–8, LR 5e-6–1e-5, KL penalty.

## Mốc 5 (Ngày 17–21): Hậu xử lý Roblox & đánh giá cuối

- [ ] `roblox_export.py` chạy cuối pipeline, theo thứ tự:
  Decimate (< 20K tris/MeshPart, mục tiêu 500–5K cho prop) → `recalculate_outside()` → đặt đáy tại Z = 0, pivot tâm đáy → scale theo stud (1 stud ≈ 0.28 m) → Smart UV Project → bake PBR (Color/Roughness/Metalness/Normal, ≤ 1024²) nếu dùng procedural shader, hoặc giữ material/vertex color sạch cho low-poly → xuất `.glb`/`.fbx`.
- [ ] Kiểm tra trục khi export (Blender Z-up → Roblox Y-up).
- [ ] Bảng điểm cuối trên 3DCodeBench: baseline zero-shot vs. SFT vs. SFT + RFT (vs. + GRPO nếu có).
- [ ] Kiểm thử 50 prop thật trong Roblox Studio: ánh sáng, normal, collision, hiệu năng.

## Nguyên tắc kiểm soát rủi ro

- Dry-run 100 mẫu trước mọi lần chạy lớn.
- Luôn lưu lưới ảnh render theo checkpoint; không tin riêng loss/reward.
- Backup adapter sau mỗi mốc/vòng để quay lui.
- Không train trên `heldout/` hay 3DCodeBench dưới bất kỳ hình thức nào.
- Chạy trên Colab chỉ cho Mốc 0 và các thử nghiệm ngắn (lưu checkpoint lên Drive); RFT/GRPO nhiều ngày nên chạy trên máy A100 thuê riêng.
- Kiểm tra điều khoản sử dụng trước khi dùng output của bất kỳ model/API nào làm dữ liệu train.
