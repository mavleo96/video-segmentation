
#!/bin/bash
set -euo pipefail

# Project configuration
# TODO: move this later
ENV_NAME="video"
DATA_ROOT="/data/video-segmentation"
CITYSCAPES_ROOT="${DATA_ROOT}/cityscapes"

export DATA_ROOT CITYSCAPES_ROOT

# Initialize Conda in the current shell
eval "$(conda shell.bash hook)"

# Create and activate environment
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

# Install dependencies
python -m pip install --upgrade \
 torch \
 torchvision \
 huggingface-hub \
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
 seaborn \
 cityscapesscripts

# Configure environment variables on activation
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"

cat > "${ACTIVATE_DIR}/video-segmentation.sh" <<EOF
export DATA_ROOT="${DATA_ROOT}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT}"
EOF

# Export environment configuration
conda env export --no-builds | grep -v "^prefix: " > env.yml

echo "Environment setup complete!"
echo "Environment: ${ENV_NAME}"
echo "Environment file: env.yml"
echo "Dataset root: ${CITYSCAPES_ROOT}"
