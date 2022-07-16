#!/bin/bash

MODEL_DIR=/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/model/tfod
TFOD_PATH=/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/tfod2/models_c2793c1/research
PIPELINE_CONFIG=/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/model/pipeline.config
OUTPUT_DIR=/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/model/tfod_exported
PROTOC=true

# change virtual environment
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
workon tf2

# move to tfod directory
cd $TFOD_PATH || exit 1

# export python path
PYTHONPATH=$(pwd):$(pwd)/slim
export PYTHONPATH=$PYTHONPATH

# create proto files if needed
if [ "$PROTOC" = true ] ; then
    protoc ./object_detection/protos/*.proto --python_out=.
fi

python3 ./object_detection/exporter_main_v2.py \
            --input_type="image_tensor" \
            --pipeline_config_path=$PIPELINE_CONFIG \
            --trained_checkpoint_dir=$MODEL_DIR \
            --output_directory=$OUTPUT_DIR
