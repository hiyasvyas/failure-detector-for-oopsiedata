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
# uv sync picked torchao 0.18, which needs torch 2.9 (ImportError: cannot import name
# 'ScalingType' from 'torch.nn.functional') when the model loads. 0.13 is the one for torch 2.8.
uv pip install "torchao==0.13.0"
# Robometer reads outputs.hidden_states[-1] for Qwen3 models. With today's Unsloth (unpinned in
# Robometer's pyproject), hidden_states comes back None ("'NoneType' object is not subscriptable").
# transformers puts last_hidden_state as the last entry of hidden_states, so reading it directly
# gives the same numbers. rbm.py already falls back to it, but its check (hasattr) is always true.
sed -i 's/if hasattr(outputs, "hidden_states") else outputs.last_hidden_state/if getattr(outputs, "hidden_states", None) is not None else outputs.last_hidden_state/' robometer/models/rbm.py

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# The same imports robometer_batch.py needs. Importing PreTrainedModel loads the part that
# failed before; "import transformers" alone doesn't.
.venv/bin/python -c "import av, h5py, imageio, torchao, transformers; from transformers import PreTrainedModel; from robometer.utils.save import load_model_from_hf; from robometer.evals.eval_server import compute_batch_outputs; from robometer.utils.setup_utils import setup_batch_collator; print('robometer imports ok, transformers', transformers.__version__, 'torchao', torchao.__version__)"
.venv/bin/python -c "from huggingface_hub import snapshot_download; print('weights in', snapshot_download('robometer/Robometer-4B'))"
echo "Setup finished."
