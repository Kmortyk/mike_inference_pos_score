import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants, signature_constants
import tensorrt as trt

TRT_PATH  = '/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/model/tensorrt'

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

trt.BuilderFlag.GPU_FALLBACK = False
trt.IBuilderConfig.default_device_type = trt.DeviceType.DLA
trt.IBuilderConfig.DLA_core = 0

with open(plan_path, 'rb') as f:
    engine_data = f.read()

engine = trt_runtime.deserialize_cuda_engine(engine_data)

saved_model_loaded = tf.saved_model.load(
    TRT_PATH,
    tags=[
        tag_constants.SERVING,
    ]
)

graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
]

print(graph_func.structured_input_signature)
print(graph_func.structured_outputs)

input_data = tf.convert_to_tensor(np.ones([1, 300, 300, 3]), dtype=tf.uint8)
print(input_data.shape)

output = graph_func(input_data)

print(output)


