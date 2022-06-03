import sys
import cv2

from src.config import configure_script # do not delete
import src.calc as calc
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from src.ros.camera_node import CameraNode

bridge = CvBridge()
fps_sum = 0
fps_mes_count = 0

st_time = time.time()

def callback(msg: Image):
    global st_time, fps_sum, fps_mes_count

    start_time = time.time()

    cv2_img = bridge.imgmsg_to_cv2(msg)

    out = calc.calc_score_image(cv2_img)

    print(time.time() - st_time)

    fps_cur = 1.0 / (time.time() - start_time)
    fps_mes_count += 1
    fps_sum += fps_cur

    print(out.nearest_score, f"FPS_mean: {fps_sum / fps_mes_count} FPS_cur: {fps_cur}")

# os.environ['ROS_PYTHON_LOG_CONFIG_FILE'] = '/home/kmortyk/.ros/log/inference_module.log'

class InferenceNode(object):
    def __init__(self):
        rospy.init_node(self.name(), anonymous=False)

    def start(self):
        rospy.Subscriber('/' + CameraNode.name(), Image, callback)
        rospy.spin()

    @staticmethod
    def name() -> str:
        return 'inference_module'

if __name__ == '__main__':
    InferenceNode().start()
