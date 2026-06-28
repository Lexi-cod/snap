#!/bin/bash
# ExecuTorch setup for Qualcomm QNN backend
# Run this when Qualcomm AI Engine Direct SDK is available

set -e

python3 -m venv ~/executorch-env
source ~/executorch-env/bin/activate

git clone -b release/1.3 https://github.com/pytorch/executorch
cd executorch
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt requirements-examples.txt
pip install -e .

echo "ExecuTorch setup complete. Next steps:"
echo "1. Download Qualcomm AI Engine Direct SDK from https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk"
echo "2. Set QNN_SDK_ROOT to SDK path"
echo "3. Run ./backends/qualcomm/scripts/build.sh"
echo "4. Export model: python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s SERIAL -m SM8750 --decoder_model qwen2_5_vl_3b --artifact ./qwen_qnn"
