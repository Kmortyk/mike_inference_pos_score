import os

BASE_PATH        = os.path.abspath("/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction")

# define model paths
CKPT_PATH        = os.path.join(BASE_PATH, "model/ckpt-251")
PIPELINE_PATH    = os.path.join(BASE_PATH, "model/pipeline.config")
PATH_TO_LABELS   = os.path.join(BASE_PATH, "model/classes.pbtxt")

# test paths
RECORD_PATH      = os.path.join(BASE_PATH, "data/testing_grayscale.record")
IMAGES_PATH      = os.path.join(BASE_PATH, "data/img")
# IMAGES_PATH      = os.path.join(BASE_PATH, "data/img_diff_pos/grayscale")

# define output configuration
MIN_SCORE_THRESH = 0.5

TFOD2_PATH      = os.path.join(BASE_PATH, "tfod2/models_c2793c1/research")
TFOD2_SLIM_PATH = os.path.join(BASE_PATH, "tfod2/models_c2793c1/research/slim")

# camera for the main_video script
CAMERA_INDEX = 0

# size of the topics queue
PUBLISH_QUEUE_SIZE = 100
