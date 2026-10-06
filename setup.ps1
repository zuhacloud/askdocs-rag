# One-time setup on Windows. Usage: .\setup.ps1
# Creates the virtual environment, installs PyTorch (CUDA if an NVIDIA GPU is present),
# installs the dependencies, creates .env, and downloads the models.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "[askdocs] $msg" -ForegroundColor Cyan }

# 1. Virtual environment (prefers Python 3.12, the most widely supported version for ML packages)
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $pyExe = "python"; $pyArgs = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $ErrorActionPreference = "Continue"  # probing versions that may not exist writes to stderr
        foreach ($v in "3.12", "3.11", "3.13") {
            $null = & py "-$v" -c "pass" 2>&1
            if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-$v"); break }
        }
        $ErrorActionPreference = "Stop"
    }
    Say "Creating virtual environment with $pyExe $pyArgs"
    & $pyExe @pyArgs -m venv .venv
}
$python = ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip --quiet

# 2. PyTorch: CUDA build when an NVIDIA GPU is present, CPU build otherwise
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Say "NVIDIA GPU detected, installing PyTorch with CUDA"
    & $python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet
} else {
    Say "No NVIDIA GPU detected, installing PyTorch for CPU"
    & $python -m pip install torch --quiet
}

# 3. Everything else
Say "Installing dependencies"
& $python -m pip install -r requirements.txt --quiet

# 4. Settings
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Say "Created .env (defaults to a local Ollama model)"
}

# 5. Download the retrieval models now, so the first start is quick
Say "Downloading retrieval models"
& $python -c "from askdocs.config import get_settings as g; from askdocs.models import Embedder, Reranker; s = g(); Embedder(s.embed_model, 'cpu'); Reranker(s.rerank_model, 'cpu')"

# 6. Language model (optional: any OpenAI-compatible API also works)
$model = (Select-String -Path ".env" -Pattern "^\s*LLM_MODEL=(.+)$" | Select-Object -First 1).Matches.Groups[1].Value
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama -and (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
    $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"  # installed, but this terminal predates the PATH update
}
if ($ollama) {
    Say "Pulling the language model ($model) with Ollama"
    & $ollama pull $model
} else {
    Say "Ollama not found. Install it from https://ollama.com (or run: winget install Ollama.Ollama),"
    Say "then run: ollama pull $model   -- or point .env at OpenAI / Groq instead."
}

Say "Setup complete. Start the app with: .\run.ps1"
