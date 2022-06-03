import itertools
import sys
import cv2

import rospy
from std_msgs.msg import Float64, Float64MultiArray, Bool, MultiArrayLayout, MultiArrayDimension
from src.config import configure_script # do not delete
from src.config import config
from src.model import CalcScoreOutput
from src.ros.fps_meter import FPSMeter
import src.calc as calc


class InferenceNode(object):
    def __init__(self):
        # init node
        rospy.init_node(self.name(), anonymous=False)
        # topics
        self.pub_scr = rospy.Publisher('inference_scores', Float64, queue_size=config.PUBLISH_QUEUE_SIZE)
        self.pub_bbx = rospy.Publisher('inference_bboxes', Float64MultiArray, queue_size=config.PUBLISH_QUEUE_SIZE)
        self.pub_flag_reached = rospy.Publisher('inference_reached', Bool, queue_size=config.PUBLISH_QUEUE_SIZE)
        # other utils
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.fps_meter = FPSMeter()
        # check camera here, before start of the node
        if not self.cap.isOpened():
            sys.exit(f"[e] cannot open camera by index '{config.CAMERA_INDEX}'. exit.")

    @staticmethod
    def name():
        return "inference_node"

    def publish_scores(self, out: CalcScoreOutput):
        msg = Float64()
        msg.data = out.nearest_score
        self.pub_scr.publish(msg)

    def publish_reached(self, out: CalcScoreOutput):
        msg = Bool()
        msg.data = out.reached_flag
        self.pub_flag_reached.publish(msg)

    def publish_bboxes(self, out: CalcScoreOutput):
        # get bboxes list
        bboxes = out.normalized_bboxes_list
        print(list(itertools.chain(*bboxes)))
        # define dimensions
        layout = MultiArrayLayout()
        layout.data_offset = 0
        layout.dim.append(MultiArrayDimension("len", len(bboxes), 4))
        layout.dim.append(MultiArrayDimension("coord", 4, 1))
        # create msg
        msg        = Float64MultiArray()
        msg.layout = layout
        msg.data   = list(itertools.chain(*bboxes))
        self.pub_bbx.publish(msg)

    def start(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                sys.exit(f"[e] can't receive frame from camera '{config.CAMERA_INDEX}'. exit.")

            with self.fps_meter:
                out = calc.calc_score_image(frame)

            # publish calculated values
            self.publish_scores(out)
            self.publish_bboxes(out)
            self.publish_reached(out)

            # print node info
            self.fps_meter.print()
            print(f"[⌛] publish bounding boxes and scores {out}.")

if __name__ == '__main__':
    print("[⌛][atm(⚛)] start inference node ...")
    InferenceNode().start()
