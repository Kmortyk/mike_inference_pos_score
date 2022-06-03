from src.config import config
from src.model import Bbox
from src.model.calc_score_output import CalcScoreOutput
from src.preprocess import *
from src.calc.detect_fn import detect_fn
import tensorflow as tf
import numpy as np

# define preprocessors
preps = [
    ResizePreprocessor(320, 320),
    GrayscalePreprocessor()
]

def calc_score_image(image_np) -> CalcScoreOutput:
    for prep in preps:
        image_np = prep.preprocess(image_np)

    tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0), dtype=tf.float32)
    detects, predictions_dict, _ = detect_fn(tensor)

    boxes = detects['detection_boxes'][0]
    confidence = detects['detection_scores'][0]
    dqn_scores = detects['dqn_scores']

    out = CalcScoreOutput(image_np)

    for (box, conf, dqn_score) in zip(boxes, confidence, dqn_scores):
        if conf >= config.MIN_SCORE_THRESH:
            box_np = box.numpy()
            (y_min, x_min, y_max, x_max) = box_np

            out.append_bbox(Bbox(x_min, y_min, x_max, y_max), dqn_score.numpy())

    return out
