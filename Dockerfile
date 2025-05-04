FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    patch \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig \
    liblmdb-dev && \
    rm -rf /var/lib/apt/lists/*

# RUN apt update && \
#     apt install -y wget libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig  python3 pip && \
#     wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb && \
#     dpkg -i libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*
# 如果 libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb 404 not found
# 请打开 http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/ 查找 libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb 对应的新版本
WORKDIR /app

COPY . .

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV PATH="/app/.venv/bin:$PATH"
ENV API_AUTH_KEY=mt_photos_ai_extra
ENV CLIP_MODEL=ViT-B-16
ENV RECOGNITION_MODEL=buffalo_l
ENV DETECTION_THRESH=0.65
ENV DEVICE=cuda
ENV CLIP_DOWNLOAD_ROOT=/app/.cache/clip

EXPOSE 8060
VOLUME [ "/app/.cache/clip", "/app/.venv/lib/python3.11/site-packages/rapidocr/models/", "/root/.insightface/models"]
CMD [ "python", "/app/main.py" ]

HEALTHCHECK CMD python3 scripts/healthcheck.py
