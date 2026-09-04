#!/usr/bin/env bash
# Fetch every model the project uses, fresh: the thesis model and the embedder from
# Hugging Face, and the Ollama tags the paper measured from the Ollama library. Pass
# --fresh to delete the cached copies first. Needs the ml extra (uv sync --extra ml).
set -euo pipefail
cd "$(dirname "$0")/.."

HF_MODELS=("ByteDance/Ouro-1.4B" "sentence-transformers/all-MiniLM-L6-v2")
# Library tags, not Hugging Face GGUFs: a GGUF is a different quantised artefact from
# the one the reported numbers were produced on, and the paper cites these tags.
OLLAMA_MODELS=("llama3.2:3b" "qwen2.5:7b" "llama3.1:8b")

if [[ "${1:-}" == "--fresh" ]]; then
  for repo in "${HF_MODELS[@]}"; do
    rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${repo//\//--}"
  done
  for tag in "${OLLAMA_MODELS[@]}"; do ollama rm "$tag" 2>/dev/null || true; done
fi

uv run python - "${HF_MODELS[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print("fetched", repo, "->", snapshot_download(repo), flush=True)
PY

if command -v ollama >/dev/null; then
  for tag in "${OLLAMA_MODELS[@]}"; do ollama pull "$tag"; done
  ollama list
else
  echo "ollama is not installed; skipped the Ollama tags" >&2
fi
