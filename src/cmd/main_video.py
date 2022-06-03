import sys
import cv2

from src.config import configure_script, config  # do not delete
import src.calc as calc
import src.draw as draw
import time

if __name__ == '__main__':
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            sys.exit("can't receive frame")

        start_time = time.time()

        out = calc.calc_score_image(frame)
        image = out.image_np

        print("FPS: ", 1.0 / (time.time() - start_time))

        draw.draw_calc_output(image, out)

        print(out.nearest_score)  # print the image score

        cv2.imshow("Video Sample", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
