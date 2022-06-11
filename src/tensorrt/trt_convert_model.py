from tensorflow.python.compiler.tensorrt import trt_convert as trt

TFOD_PATH = '/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/model/tfod_exported/saved_model'
TRT_PATH  = '/mnt/sda1/Projects/PycharmProjects/DQNScoresFunction/model/tensorrt'

if __name__ == '__main__':
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=TFOD_PATH)
    converter.convert()
    converter.save(TRT_PATH)
