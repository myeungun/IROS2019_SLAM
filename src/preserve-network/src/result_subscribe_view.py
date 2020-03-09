#!/usr/bin/env python

##### DEFINITIONS

#### ZED resolution
### RESOLUTION_HD2K
#CAM_WIDTH = 2208
#CAM_HEIGHT = 1242
#CAM_CHANNELS = 3
### RESOLUTION_HD1280
#CAM_WIDTH = 1920
#CAM_HEIGHT = 1080
#CAM_CHANNELS = 3
### RESOLUTION_HD720
#CAM_WIDTH = 1280
#CAM_HEIGHT = 720
#CAM_CHANNELS = 3
### RESOLUTION_VGA
CAM_WIDTH = 672
CAM_HEIGHT = 376
CAM_CHANNELS = 3
#######
WINDOW_NAME_RESULT = 'Result'
###
from operator import mul

IMG_SHAPE_ZED = (CAM_HEIGHT, CAM_WIDTH, CAM_CHANNELS)
IMG_SIZE_ZED = reduce(mul, IMG_SHAPE_ZED)
##
import sys
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
###

class Result_From_TX2:

	def __init__(self):
		self.bridge = CvBridge()
		self.networkResult_sub = rospy.Subscriber('/preserve_network/resultImage', Image, self.resultViewCallback)

	def resultViewCallback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imshow(WINDOW_NAME_RESULT, cv_image)
		cv2.waitKey(1)

	def unsubscribe_imageFromZED(self):
		self.networkResult_sub.unregister()

def main(args):
	rospy.init_node('network_result_subscriber')
	
	ic = Result_From_TX2()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	###
	print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
