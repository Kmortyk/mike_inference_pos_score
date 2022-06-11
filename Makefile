# install all the dependencies
install-deps:
	pip3 install -r requirements.txt

# run inference on the sample images
# default images directory: "data/img"
run-image:
	./scripts/run-image.sh

# run inference on stream from the camera
run-video:
	./scripts/run-video.sh

# run ros node with publishing values to topics
run-ros:
	./scripts/run-ros.sh

docker-build:
	docker build . -t km/dqn_scores_function:1.0

# docker run -u $(id -u):$(id -g) -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter jupyter notebook --ip=0.0.0.0
docker-run:
	docker run --gpus all --device="/dev/video1:/dev/video0" km/dqn_scores_function:1.0