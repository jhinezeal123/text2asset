#!/usr/bin/env bash
# Tải bản dựng nari-llama.cpp (CUDA 12.8) + Qwen3.8-27B NariQ 3-bit GGUF + mmproj. In BIN_READY / MODEL_READY.
set -e
mkdir -p /content/nari && cd /content/nari
R=https://github.com/nari-labs/nari-llama.cpp/releases/download/b12431-nariv0
for f in nari-llama.cpp-b12431-nariv0-bin-linux-cuda-12.8-x64.tar.gz cudart-nari-llama.cpp-b12431-nariv0-linux-cuda-12.8-x64.tar.gz SHA256SUMS; do
  [ -f "$f" ] || curl -sSLO "$R/$f"
done
grep linux SHA256SUMS | sha256sum -c -
mkdir -p bin && for t in *.tar.gz; do tar -xzf "$t" -C bin; done
echo BIN_READY
python3 -c "from huggingface_hub import hf_hub_download as d; [print(d('nari-labs/Qwen3.8-27B-NariQ-GGUF', f, local_dir='/content/nari/model')) for f in ['Qwen3.8-27B-NariQ-3.0bpw.gguf','Qwen3.8-27B-NariQ-mmproj-BF16.gguf']]"
echo MODEL_READY
