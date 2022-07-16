import itertools
import sys
import time

import rospy
from std_msgs.msg import Float64, Float64MultiArray, Bool, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image
from src.config import configure_script  # do not delete
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
        rospy.Subscriber(config.CAMERA_TOPIC, Image, self.img_callback)
        self.pub_scr = rospy.Publisher('inference_scores', Float64MultiArray, queue_size=config.PUBLISH_QUEUE_SIZE)
        self.pub_bbx = rospy.Publisher('inference_bboxes', Float64MultiArray, queue_size=config.PUBLISH_QUEUE_SIZE)
        self.pub_flag_reached = rospy.Publisher('inference_reached', Bool, queue_size=config.PUBLISH_QUEUE_SIZE)
        # other utils
        self.fps_meter = FPSMeter()
        self.cv_bridge = CvBridge()
        # nodes
        self.frame = None

    @staticmethod
    def name():
        return "inference_node"

    def img_callback(self, image_data):
        self.frame = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)

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
        msg = Float64MultiArray()
        msg.layout = layout
        msg.data = list(itertools.chain(*bboxes))
        self.pub_bbx.publish(msg)

    def start(self):
        while True:
            if self.frame is None:
                print("[⌛][atm(⚛)] got no frame. trying again ...")
                time.sleep(0.1)
                continue

            with self.fps_meter:
                f = self.frame
                self.frame = None

                out = calc.calc_score_image(f)

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
