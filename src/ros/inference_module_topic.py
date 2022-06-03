import sys
import cv2
import rospy

from src.config import configure_script, config
import src.calc as calc

from src.ros.fps_meter import FPSMeter
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayDimension

QUEUE_SIZE = 10

class InferenceNode:
    def __init__(self):
        rospy.init_node(self.name(), anonymous=False, log_level=rospy.INFO)

        self.fps_meter = FPSMeter()

        self.loop_rate = rospy.Rate(100)
        self.pub_score = rospy.Publisher(self.name() + '/score', Float64, queue_size=QUEUE_SIZE)
        self.pub_bboxes = rospy.Publisher(self.name() + '/bboxes', Float32MultiArray, queue_size=QUEUE_SIZE)

    @staticmethod
    def name():
        return 'inference_module'

    def start(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            sys.exit("cannot open camera")

        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                sys.exit("can't receive frame")

            with self.fps_meter:
                out = calc.calc_score_image(frame)

            print(f"score: {out.nearest_score:.2f} "
                  f"FPS_mean: {self.fps_meter.mean():.2f} "
                  f"FPS_cur: {self.fps_meter.last():.2f}")

            self.pub_score.publish(out.nearest_score)

            if len(out.normalized_bboxes) > 0:
                for bbox in out.normalized_bboxes_list:
                    arr = Float32MultiArray()
                    arr.layout.dim.append(MultiArrayDimension())
                    arr.layout.dim[0].size = 4
                    arr.data = bbox

                    print(out.normalized_bboxes_list)
                    self.pub_bboxes.publish(arr)

            self.loop_rate.sleep()

if __name__ == '__main__':
    InferenceNode().start()
