import math
from typing import List

from src.model import Bbox, EMPTY_BBOX

REACHED_THRESH = 40 / 100.

class CalcScoreOutput:
    def __init__(self, image_np):
        self.image_np = image_np
        self.normalized_bboxes: List[Bbox] = [] # in range [0..1], [0..1]
        self.normalized_bboxes_list: List[List[float]] = [] # in range [0..1], [0..1]
        self.image_score: float = 0
        self.nearest_bbox: Bbox = EMPTY_BBOX
        self.max_area: float = -math.inf
        self.nearest_score: float = 0
        self.bboxes_scores: List[float] = []
        self.reached_flag: bool = False

    def object_found(self) -> bool:
        return len(self.normalized_bboxes) > 0

    def append_bbox(self, bbox, bbox_score):
        bbox_area = bbox.area()
        # find nearest_bbox with good confidence
        if bbox_area > self.max_area:
            self.max_area = bbox_area
            self.nearest_bbox = bbox
            self.nearest_score = bbox_score
        # check whether destination is reached
        if self.max_area >= REACHED_THRESH:
            self.reached_flag = True
        self.normalized_bboxes.append(bbox)
        self.normalized_bboxes_list.append(bbox.list())
        self.image_score += bbox_score
        self.bboxes_scores.append(bbox_score)

    def __str__(self) -> str:
        return f"[score={self.nearest_score:.02f}," \
               f"reached={self.reached_flag}," \
               f"area={self.max_area:.02f}," \
               f"bboxes={self.normalized_bboxes}]"
