#!/bin/bash
set -euo pipefail

ENV_NAME="video"

conda create -n $ENV_NAME python=3.12 -y && \
conda init bash && \
eval "$(conda shell.bash hook)" && \
conda activate $ENV_NAME && \
pip install -U torch \
 torchvision \
 huggingface-hub \
 hf-cli \
 transformers \
 accelerate \
 datasets \
 ipykernel \
 wandb \
 ruff \
 tqdm \
 pre-commit \
 tensorboard \
 scipy \
 numpy \
 pandas \
 matplotlib \
 seaborn

conda env export --no-builds | grep -v "^prefix: " > env.yml