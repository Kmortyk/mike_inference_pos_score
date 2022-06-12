import sys
from datetime import datetime
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from src.config import config


class CameraNode(object):
    def __init__(self, camera_idx):
        rospy.init_node(CameraNode.name(), anonymous=False, log_level=rospy.INFO)
        print(f"[⌛] open device {camera_idx} ...")
        self.cap = cv2.VideoCapture(camera_idx)
        self.bridge    = CvBridge()
        self.loop_rate = rospy.Rate(100)
        self.pub = rospy.Publisher(CameraNode.name() + '/raw', Image, queue_size=10)

    def start(self):
        print(f"[⌛] start main loop ...")
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                sys.exit("can't receive frame")

            print(datetime.now().strftime("%H:%M:%S"), 'publish image')

            self.pub.publish(self.bridge.cv2_to_imgmsg(frame))

            self.loop_rate.sleep()

    @staticmethod
    def name() -> str:
        return 'mike_camera'

if __name__ == '__main__':
    print("[⌛] start camera node ...")
    CameraNode(config.CAMERA_INDEX).start()
