ARG DEVICE=cuda

#region builder
FROM python:3.11-bookworm AS builder-cpu
FROM builder-cpu AS builder-cuda

FROM builder-${DEVICE} AS builder
ARG DEVICE
WORKDIR /app

# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# COPY . .

# # Install dependencies
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --extra ${DEVICE} --frozen --no-dev --compile --link-mode copy
#endregion builder


FROM python:3.11-slim-bookworm AS prod-cpu
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS prod-cuda


#region prod
FROM prod-${DEVICE} AS prod
ARG DEVICE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    patch \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig \
    liblmdb-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY . .
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra ${DEVICE} --frozen --no-dev --compile --link-mode copy


# ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    API_AUTH_KEY=mt_photos_ai_extra \
    CLIP_MODEL=ViT-B-16 \
    RECOGNITION_MODEL=buffalo_l \
    DETECTION_THRESH=0.65 \
    DEVICE=${DEVICE} \
    CLIP_DOWNLOAD_ROOT=/app/.cache/clip

EXPOSE 8060
VOLUME [ "/app/.cache/clip", "/app/.venv/lib/python3.11/site-packages/rapidocr/models/", "/root/.insightface/models"]
CMD [ "python", "-m", "app" ]

HEALTHCHECK CMD python3 scripts/healthcheck.py
#endregion prod
