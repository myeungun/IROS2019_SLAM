#!/usr/bin/env python

##### DEFINITIONS

#### ZED resolution
### RESOLUTION_HD2K
#CAM_WIDTH = 2208
#CAM_HEIGHT = 1242
#CAM_CHANNELS = 1
### RESOLUTION_HD1280
#CAM_WIDTH = 1920
#CAM_HEIGHT = 1080
#CAM_CHANNELS = 1
### RESOLUTION_HD720
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_CHANNELS = 1
### RESOLUTION_VGA
#CAM_WIDTH = 672
#CAM_HEIGHT = 376
#CAM_CHANNELS = 1
#######
WINDOW_NAME_DEPTH = 'DEPTH_IMAGE'
###
from operator import mul

IMG_SHAPE_DEPTH = (CAM_HEIGHT, CAM_WIDTH, CAM_CHANNELS)
IMG_SIZE_DEPTH = reduce(mul, IMG_SHAPE_DEPTH)
##
import sys
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
###

class DEPTH_IMAGE:

	def __init__(self):
		self.bridge = CvBridge()
		self.depthImage_sub = rospy.Subscriber('/zed/depth/depth_registered', Image, self.depthViewCallback)
                self.depthImage_pub = rospy.Publisher('/hr/depth/depth_registered', Image, queue_size=10)

	def depthViewCallback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                        depth_array = np.array(cv_image, dtype=np.float32)
                        self.depthImage_pub.publish(self.bridge.cv2_to_imgmsg(depth_array, "32FC1"))

		except CvBridgeError as e:
			print(e)
		
	def unsubscribe_DepthImage(self):
		self.depthImage_sub.unregister()

def main(args):
	rospy.init_node('depthImage_Subscriber')
	
	ic = DEPTH_IMAGE()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	###
	print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
