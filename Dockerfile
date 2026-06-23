# syntax=docker/dockerfile:1
#
# SatsangAI V1 — self-contained app image.
#   build:  docker build -t satsangai .
#   run:    docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... satsangai
#   then:   open http://localhost:8000/   (chat UI),  GET /health
#
# Self-contained: the prebuilt counseling-core index is copied in, and the two
# BGE retrieval models (embedder + reranker) are baked at build time, so the
# container starts fully offline (HF_HUB_OFFLINE=1) with no first-request stall.
# Only ANTHROPIC_API_KEY (for generation/planning) is supplied at runtime.
#
# The KB (../satsangai) is NOT needed at runtime — it's only used by
# `python -m api.build_index` to regenerate api/data/counseling_index.parquet.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SATSANG_EMBED_DEVICE=cpu

WORKDIR /app

# libgomp1: OpenMP runtime needed by the torch/sentence-transformers CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so sentence-transformers doesn't drag in the CUDA wheel.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the retrieval models into the image (both public — no HF token needed).
# Set BAKE_MODELS=0 to skip and instead download on first request (smaller image,
# but then mount a persistent /models volume and drop HF_HUB_OFFLINE).
ARG BAKE_MODELS=1
RUN if [ "$BAKE_MODELS" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"; \
    fi

# App code + the prebuilt counseling-core index (api/data/counseling_index.parquet).
# config/ + enrichment/ are included so `python -m api.build_index` can run in-container
# if the KB is mounted; they are not imported by the request path.
COPY api ./api
COPY config ./config
COPY enrichment ./enrichment

# Already-cached models -> run offline; only the Anthropic API is reached at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
