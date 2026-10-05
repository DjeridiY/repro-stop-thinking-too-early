#!/usr/bin/env bash
# On the Colab VM: authors' looped-model environment + fp16 patch, then one training run in the background.
# Usage: vm_setup.sh "<e71_ouro_map.py arguments>"
set -euo pipefail
cd /content/w/upstream
patch -p1 -N < ../patches/fp16.patch
uv venv -q --python 3.11 .venv-loop
uv pip install -q --python .venv-loop/bin/python -r requirements-looped.txt
mkdir -p results
{ date -u; nvidia-smi -L; uv pip freeze --python .venv-loop/bin/python; } > results/environment.txt
cd scripts
nohup ../.venv-loop/bin/python -u e71_ouro_map.py $1 > ../results/train.log 2>&1 &
echo "started: e71_ouro_map.py $1"
