import cv2


class MeanPreprocessor:
    def __init__(self, r_mean=0, g_mean=0, b_mean=0, means=None):
        # store the Red, Green, and Blue channel averages across a training set
        if means is not None:
            self.r_mean = means["R"]
            self.g_mean = means["G"]
            self.b_mean = means["B"]
        else:
            self.r_mean = r_mean
            self.g_mean = g_mean
            self.b_mean = b_mean

    def preprocess(self, image):
        # split the image into its respective Red, Green, and Blue channels
        # float32 for negative values
        (B, G, R) = cv2.split(image.astype("float32"))
        # subtract the means for each channel
        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean
        # merge the channels back together and return the image
        return cv2.merge([B, G, R])
