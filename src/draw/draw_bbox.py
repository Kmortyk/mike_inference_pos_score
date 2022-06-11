import cv2

from src.model import Bbox
from src.model import CalcScoreOutput

COLOR         = (172, 217, 153) # acd999
COLOR_NEAREST = (0, 0, 255) # red

def draw_calc_output(image, out: CalcScoreOutput):
    for bbox, score in zip(out.normalized_bboxes, out.bboxes_scores):
        draw_bbox(image, bbox, score=score, label="score")
    draw_bbox(image, out.nearest_bbox, score=out.nearest_score, color=COLOR_NEAREST, label="score")

def draw_bbox(dst, bbox: Bbox, score=0.5, label="object", color=COLOR):
    height, width = dst.shape[:2]

    # scale the bounding box from the range [0,1] to [0, width], [0, height]
    min_x = int(bbox.x_min * width)
    min_y = int(bbox.y_min * height)
    max_x = int(bbox.x_max * width)
    max_y = int(bbox.y_max * height)

    # draw the prediction on the output image
    label = f"{label}: {score}"
    cv2.rectangle(dst, (min_x, min_y), (max_x, max_y), color, 2)
    y = min_y - 10 if min_y - 10 > 10 else min_y + 10

    dy = 8
    for i, line in enumerate(label.split('\n')):
        y = y + i * dy
        cv2.putText(dst, line, (min_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
