class NormalizePreprocessor:
    def __init__(self, dtype="float"):
        self.dtype = dtype

    def preprocess(self, image):
        return image.astype(self.dtype) / 255.0
