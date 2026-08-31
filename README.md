# Video Segmentation

## Quick cmds

```bash
git clone git@github.com:mavleo96/video-segmentation.git
cd video-segmentation


# create the conda env
bash scripts/env.sh
conda activate video

# install the git hook
pre-commit install

# download cityscapes ($DATA_ROOT and $CITYSCAPES_ROOT come from the env)
tmux new -s dl
conda activate video
cd video-segmentation
bash scripts/download_cityscapes.sh --splits train,val --workers 32
bash scripts/download_kittistep.sh

# build the dataset indexes (-> $CITYSCAPES_ROOT/index, $KITTISTEP_ROOT/index)
python -m src.data.index
python -m src.data.index --dataset kitti_step

# precompute RAFT flow (-> $CITYSCAPES_ROOT/flow/{train,val}; ~31 GB / ~61 GB)
python -m scripts.precompute_flow_train --coverage   # optional: FB coverage per offset, saves nothing
python -m scripts.precompute_flow_train              # labeled frame <-> ±1..±5, 1/4 res
python -m scripts.precompute_flow_val                # consecutive frames, 1/2 res

# ruff through pre-commit hooks
pre-commit run --all-files

```
