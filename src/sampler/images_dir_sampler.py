import cv2
from imutils import paths
import random

class ImagesDirSampler:
    def __init__(self, images_dir_path):
        self.image_paths = list(paths.list_images(images_dir_path))
        self.cur_image_idx = 0

        random.shuffle(self.image_paths)

    def take_count(self, count):
        images = []

        for i in range(count):
            images.append(self.take())

        return images

    def take(self):
        if self.cur_image_idx == len(self.image_paths):
            raise StopIteration

        image = cv2.imread(self.image_paths[self.cur_image_idx])
        self.cur_image_idx += 1

        return image
