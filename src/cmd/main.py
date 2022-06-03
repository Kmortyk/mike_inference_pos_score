import cv2

from src.config import configure_script, config # do not delete
from src.sampler import ImagesDirSampler
import src.calc as calc
import src.draw as draw

TAKE_IMAGES = 15
# TAKE_IMAGES = 116

if __name__ == '__main__':
    sampler = ImagesDirSampler(config.IMAGES_PATH)
    images = sampler.take_count(TAKE_IMAGES)

    for image in images:
        out = calc.calc_score_image(image)
        image = out.image_np

        draw.draw_calc_output(image, out)

        print(out.nearest_score) # print the image score

        cv2.imshow("Sample", image)
        cv2.waitKey(0)
