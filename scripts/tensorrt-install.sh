#!/bin/bash

# docs: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

workon "DQNScoresFunction-tf-2-3"

TRT_PATH=/home/kmortyk/Sources/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0

cd "$TRT_PATH"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TRT_PATH"

cd ./python       && python3 -m pip install tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
cd ./uff          && python3 -m pip install uff-0.6.9-py2.py3-none-any.whl
cd ./graphsurgeon && python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl