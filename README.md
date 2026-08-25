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

# ruff commands
ruff check --fix .
ruff format .
pre-commit run --all-files
```
