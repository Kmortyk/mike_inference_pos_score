import cv2

class GrayscalePreprocessor:
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.merge([gray, gray, gray])
