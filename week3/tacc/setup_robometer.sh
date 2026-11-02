#!/bin/bash
# One-time Robometer install on Lonestar6. Run it on a GPU dev node, not on the login node:
#   idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00
#   bash $WORK/failure-detector-for-oopsiedata/week3/tacc/setup_robometer.sh
# Installs into $WORK/robometer/.venv. Nothing is trained.
set -euo pipefail

# Big downloads go to $SCRATCH. $HOME only holds 10 GB.
export UV_CACHE_DIR=$SCRATCH/uv-cache
export HF_HOME=$SCRATCH/hf-cache
export UV_PYTHON_INSTALL_DIR=$WORK/uv-python
export PATH=$HOME/.local/bin:$PATH

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd $WORK
if [ ! -d robometer ]; then
  git clone https://github.com/robometer/robometer.git
fi
cd robometer

# Robometer asks for Python 3.10; uv downloads it. The "robometer" extra adds transformers.
uv sync --extra robometer
# Episode files and video readers for our week-2 loader.
uv pip install h5py imageio av imageio-ffmpeg

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -c "import robometer, transformers, av, h5py, imageio; print('robometer imports ok, transformers', transformers.__version__)"
.venv/bin/python -c "from huggingface_hub import snapshot_download; print('weights in', snapshot_download('robometer/Robometer-4B'))"
echo "Setup finished."
