import sys

from src.preprocess import ResizePreprocessor

sys.path.append("../../../src/inference/")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tfod2.models_c2793c1.research.object_detection.utils import label_map_util
from tfod2.models_c2793c1.research.object_detection.utils import config_util
from tfod2.models_c2793c1.research.object_detection.utils import visualization_utils as viz_utils
from tfod2.models_c2793c1.research.object_detection.builders import model_builder
tf.get_logger().setLevel('ERROR')
import cv2
import numpy as np

PATH_TO_CFG = "/mnt/sda1/Projects/PycharmProjects/MikeHotel_TFOD2/model/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config"
PATH_TO_CKPT = "/mnt/sda1/Projects/PycharmProjects/MikeHotel_TFOD2/model/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint"
PATH_TO_LABELS = "/mnt/sda1/Projects/PycharmProjects/MikeHotel_TFOD2/dataset/bottle_soup/classes.pbtxt"

# enable gpu map growing
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load model config and build model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# load model checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# convert labels to indices
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# create video capture and preprocessor
cap = cv2.VideoCapture(0)
prep = ResizePreprocessor(300, 300)

# start pipeline
while True:
    # get frame from camera
    ret, image_np = cap.read()
    image_np = prep.preprocess(image_np)
    image_np_with_detections = image_np.copy()

    # expand image dimensions and convert tot tensor
    image_np_expanded = np.expand_dims(image_np, axis=0)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    # get model predictions
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    # draw detections on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + 1).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    # show image with detections
    cv2.imshow('object detection', image_np_with_detections)

    # wait for key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
