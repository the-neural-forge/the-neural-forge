#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade # install torch 2.6.0
python data_speedrun/cached_fineweb10B.py 10