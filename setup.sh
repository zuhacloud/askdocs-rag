#!/usr/bin/env bash
# One-time setup on macOS / Linux. Usage: bash setup.sh
# Creates the virtual environment, installs PyTorch (CUDA if an NVIDIA GPU is present),
# installs the dependencies, creates .env, and downloads the models.
set -euo pipefail
cd "$(dirname "$0")"
say() { printf '\033[36m[askdocs]\033[0m %s\n' "$1"; }

# 1. Virtual environment (prefers Python 3.12)
if [ ! -x .venv/bin/python ]; then
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
  say "Creating virtual environment with $PY"
  "$PY" -m venv .venv
fi
PYTHON=.venv/bin/python
"$PYTHON" -m pip install --upgrade pip --quiet

# 2. PyTorch: CUDA build on Linux with an NVIDIA GPU, default build otherwise (Apple Silicon included)
if command -v nvidia-smi >/dev/null 2>&1; then
  say "NVIDIA GPU detected, installing PyTorch with CUDA"
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet
else
  say "Installing PyTorch"
  "$PYTHON" -m pip install torch --quiet
fi

# 3. Everything else
say "Installing dependencies"
"$PYTHON" -m pip install -r requirements.txt --quiet

# 4. Settings
if [ ! -f .env ]; then
  cp .env.example .env
  say "Created .env (defaults to a local Ollama model)"
fi

# 5. Download the retrieval models now, so the first start is quick
say "Downloading retrieval models"
"$PYTHON" -c "from askdocs.config import get_settings as g; from askdocs.models import Embedder, Reranker; s = g(); Embedder(s.embed_model, 'cpu'); Reranker(s.rerank_model, 'cpu')"

# 6. Language model (optional: any OpenAI-compatible API also works)
MODEL=$(grep -E '^\s*LLM_MODEL=' .env | head -1 | cut -d= -f2-)
if command -v ollama >/dev/null 2>&1; then
  say "Pulling the language model ($MODEL) with Ollama"
  ollama pull "$MODEL"
else
  say "Ollama not found. Install it from https://ollama.com, then run: ollama pull $MODEL"
  say "(or point .env at OpenAI / Groq instead)"
fi

say "Setup complete. Start the app with: bash run.sh"
