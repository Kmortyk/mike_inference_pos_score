#!/bin/bash

# change path to this
MODEL=/mnt/sda1/Projects/PycharmProjects/MikeHotel_TFOD2/model/ssd_mobilenet_v2_320x320_coco17_tpu-8_grayscale

# change virtual environment
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
workon tf2

tensorboard --logdir=$MODEL