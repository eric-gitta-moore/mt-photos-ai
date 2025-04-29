FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
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

RUN uv sync --locked --extra cuda

ENV PATH="/app/.venv/bin:$PATH"
ENV API_AUTH_KEY=mt_photos_ai_extra
ENV CLIP_MODEL=ViT-B-16

EXPOSE 8060
VOLUME [ "/root/models" ]
CMD [ "python", "/app/server.py" ]
