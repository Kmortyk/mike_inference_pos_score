#!/bin/bash

# change path to this
MODEL=/mnt/sda1/Projects/PycharmProjects/MikeHotel_TFOD2/model/ssd_mobilenet_v2_320x320_coco17_tpu-8_grayscale

# initialize variables
TFOD_PATH=/mnt/sda1/Projects/PycharmProjects/MikeHotel_TFOD2/tfod2/models_c2793c1/research
PIPELINE_CONFIG=$MODEL/pipeline.config
MODEL_DIR=$MODEL/model_dir
PROTOC=false

# change virtual environment
export TF_FORCE_GPU_ALLOW_GROWTH=true
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

# start model training
python3 ./object_detection/model_main_tf2.py \
            --model_dir=$MODEL_DIR \
            --pipeline_config_path=$PIPELINE_CONFIG \
            --checkpoint_every_n=200 \
            --sample_1_of_n_eval_examples=1