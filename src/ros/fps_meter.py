import time

from dataclasses import dataclass

@dataclass
class FPSMeter:
    fps_sum:       float = 0
    fps_mes_count: float = 0
    fps_last:      float = 0

    start_time:    float = 0
    end_time:      float = 0

    def start_measure(self) -> float:
        self.start_time = time.time()
        return self.start_time

    def end_measure(self):
        self.fps_last = 1.0 / (time.time() - self.start_time)
        self.fps_sum += self.fps_last
        self.fps_mes_count += 1

    def mean(self) -> float:
        return self.fps_sum / self.fps_mes_count

    def last(self):
        return self.fps_last

    def __enter__(self):
        return self.start_measure()

    def __exit__(self, typ, value, traceback):
        self.end_measure()

    def __str__(self):
        return f"fps_mean={self.mean():.2f}, fps_cur={self.last():.2f}"

    def print(self):
        print(f"[⌚] {self}")
