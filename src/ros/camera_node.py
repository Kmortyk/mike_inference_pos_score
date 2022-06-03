import sys
from datetime import datetime
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(object):
    def __init__(self, camera_idx):
        rospy.init_node(self.name(), anonymous=False, log_level=rospy.INFO)

        self.cap = cv2.VideoCapture(camera_idx)

        self.bridge    = CvBridge()
        self.loop_rate = rospy.Rate(100)

        self.pub = rospy.Publisher(self.name(), Image, queue_size=10)

    def start(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                sys.exit("can't receive frame")

            print(datetime.now().strftime("%H:%M:%S"), 'publish image')

            self.pub.publish(self.bridge.cv2_to_imgmsg(frame))

            self.loop_rate.sleep()

    @staticmethod
    def name() -> str:
        return 'image_module'

if __name__ == '__main__':
    CameraNode(0).start()
