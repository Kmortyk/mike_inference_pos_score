import itertools
import sys
import cv2

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Float64MultiArray, Bool, MultiArrayLayout, MultiArrayDimension
from src.config import configure_script # do not delete
from src.config import config
from src.model import CalcScoreOutput
from src.model.calc_score_output import DqnScore
from src.ros.fps_meter import FPSMeter
import src.calc as calc
import numpy as np

class InferenceNode(object):
    def __init__(self):
        # init node
        rospy.init_node(self.name(), anonymous=False)
        # topics
        self.pub_scr = rospy.Publisher('inference_scores', Float64MultiArray, queue_size=config.PUBLISH_QUEUE_SIZE)
        self.pub_bbx = rospy.Publisher('inference_bboxes', Float64MultiArray, queue_size=config.PUBLISH_QUEUE_SIZE)
        self.pub_flag_reached = rospy.Publisher('inference_reached', Bool, queue_size=config.PUBLISH_QUEUE_SIZE)
        # other utils
        self.fps_meter = FPSMeter()

    @staticmethod
    def name():
        return "inference_node"

    def publish_scores(self, out: CalcScoreOutput):
        # get bboxes list
        dqn_score: DqnScore = out.nearest_score
        # define dimensions
        layout = MultiArrayLayout()
        layout.data_offset = 0
        layout.dim.append(MultiArrayDimension("components[area,dst_sq]", 2, 1))
        # create msg
        msg = Float64MultiArray()
        msg.layout = layout
        msg.data = [dqn_score.dqn_scores_area, dqn_score.dqn_scores_dst_sq]
        self.pub_scr.publish(msg)

    def publish_reached(self, out: CalcScoreOutput):
        msg = Bool()
        msg.data = out.reached_flag
        self.pub_flag_reached.publish(msg)

    def publish_bboxes(self, out: CalcScoreOutput):
        # get bboxes list
        bboxes = out.normalized_bboxes_list
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
        rospy.Subscriber('/mike_camera/raw', Image, self.callback_factory())
        rospy.spin()

    def callback_factory(self):
        def callback(msg: Image):
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

            with self.fps_meter:
                out = calc.calc_score_image(frame)

            # publish calculated values
            self.publish_scores(out)
            self.publish_bboxes(out)
            self.publish_reached(out)

            # print node info
            self.fps_meter.print()
            print(f"[⌛] publish bounding boxes and scores {out}.")

        return callback

if __name__ == '__main__':
    print("[⌛][sep(⚛)] start inference node ...")
    InferenceNode().start()
