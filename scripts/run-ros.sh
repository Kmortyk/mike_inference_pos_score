#!/bin/bash

# enable fast-fail
set -Eeo pipefail

# this is absolute path of the script directory
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
project_dir="$(realpath "$script_dir"/..)"

export PYTHONPATH="$PYTHONPATH:$project_dir:$project_dir/tfod2/models_c2793c1/research:$project_dir/tfod2/models_c2793c1/research/slim"

# start the inference
python3 "$script_dir"/../src/ros/inference_module_atm.py
