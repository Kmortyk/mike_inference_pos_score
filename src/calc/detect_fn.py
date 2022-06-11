import sys
sys.path.append("../../tfod2/models/research/object_detection")
import tensorflow as tf
from src.config import config
from tfod2.models_c2793c1.research.object_detection.utils import label_map_util
from tfod2.models_c2793c1.research.object_detection.utils import config_util
from tfod2.models_c2793c1.research.object_detection.builders import model_builder

# enable growth for all of the gpus
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load config files
configs = config_util.get_configs_from_pipeline_file(config.PIPELINE_PATH)
model_config = configs['model']
category_index = label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)

# build model from config
detection_model = model_builder.build(model_config=model_config, is_training=False)

# load weights to the built model
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(config.CKPT_PATH).expect_partial()

# this is the main prediction function, that performs on each of the images
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)

    detects = detection_model.postprocess(prediction_dict, shapes)

    return detects, prediction_dict, tf.reshape(shapes, [-1])
