# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tolerate slow or unstable connections: wait longer per request and retry failed downloads.
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# The cache mounts keep downloaded packages between builds (outside the image, so it stays small).
# If a build is interrupted, the next one reuses everything already downloaded.

# CPU-only PyTorch keeps the image small (~2 GB lighter). For GPU inference use an nvidia/cuda base image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY askdocs ./askdocs
COPY app ./app
COPY sample_docs ./sample_docs
COPY .streamlit ./.streamlit

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "askdocs.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
