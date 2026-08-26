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

# ruff commands
ruff check --fix .
ruff format .
pre-commit run --all-files

```
