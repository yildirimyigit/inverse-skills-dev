FROM mambaorg/micromamba:2.3.0-cuda12.1.1-ubuntu22.04

USER root
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROJECT_ROOT=/workspace/inverse-skills-dev \
    PYTHONPATH=/workspace/inverse-skills-dev/src \
    ENV_NAME=inverse-skills

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    pkg-config \
    tmux \
    htop \
    nano \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba create -y -n inverse-skills -f /tmp/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /workspace/inverse-skills-dev

RUN python -m pip install --upgrade pip

COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/inverse-skills-dev

RUN python -m pip install -e .

RUN mkdir -p \
    /workspace/inverse-skills-dev/artifacts \
    /workspace/inverse-skills-dev/checkpoints \
    /workspace/inverse-skills-dev/logs \
    /workspace/inverse-skills-dev/data && \
    chown -R $MAMBA_USER:$MAMBA_USER /workspace/inverse-skills-dev

USER $MAMBA_USER

CMD ["bash"]
