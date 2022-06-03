import math

class Bbox:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def area(self) -> float:
        dx = math.fabs(self.x_max - self.x_min)
        dy = math.fabs(self.y_max - self.y_min)

        return dx*dy

    def list(self):
        return [float(self.x_min), float(self.y_min), float(self.x_max), float(self.y_max)]

    def __repr__(self):
        return f"Bbox{{" \
               f"x_min={self.x_min:.02f}," \
               f"y_min={self.y_min:.02f}," \
               f"x_max={self.x_max:.02f}," \
               f"y_max={self.y_max:.02f}" \
               f"}}"

EMPTY_BBOX = Bbox(0, 0, 0, 0)
