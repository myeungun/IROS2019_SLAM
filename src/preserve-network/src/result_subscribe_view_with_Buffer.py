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

from modules.ringbuff import RingBuffer
import threading

### Buffer Initialization
BUFF_SIZE = 30
RESULT_BUFF_SIZE = IMG_SIZE_ZED * BUFF_SIZE

result_buffer = RingBuffer(RESULT_BUFF_SIZE, 'float32')
result_buffer_lock = threading.RLock()
BUFF_COUNT = 0

### Class for subscribe, convert, and view
class Result_From_TX2:

	def __init__(self):
		self.bridge = CvBridge()
		self.networkResult_sub = rospy.Subscriber('/preserve_network/result_only_Network', Image, self.resultPushBuffer)

	def resultPushBuffer(self, msg):
		global BUFF_COUNT

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		with result_buffer_lock:
			result_buffer.push(cv_image.flatten().astype('float32'))
			BUFF_COUNT += 1
			print('The number of images buffered: ', BUFF_COUNT)
			

	def resultViewCallback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imshow(WINDOW_NAME_RESULT, cv_image)
		cv2.waitKey(1)

	def unsubscribe_imageFromZED(self):
		self.networkResult_sub.unregister()

### Threading Functions
def resultView():
	global BUFF_COUNT
	global view_thread_running
	
	while view_thread_running:
		if result_buffer.nb_data > 0:
			with result_buffer_lock:
				popped_image = result_buffer.get(IMG_SIZE_ZED)
				result_buffer.pop(IMG_SIZE_ZED)
			popped_image = popped_image.reshape(IMG_SHAPE_ZED).astype('uint8')
			cv2.imshow(WINDOW_NAME_RESULT, popped_image)
			cv2.waitKey(1)
			BUFF_COUNT -= 1
			print('The number of images buffered: ', BUFF_COUNT)
			

### Node initialization
rospy.init_node('network_result_subscriber')
Result_obj = Result_From_TX2()

### Threading start
view_thread_running = True

if view_thread_running == True:
	view_thread = threading.Thread(target=resultView)
	view_thread.start()

### Start ros subscribe node
try:
	rospy.spin()
except KeyboardInterrupt:
	print("Shutting down")

###
print("Shutting down")
if view_thread_running == True:
	view_thread_running = False
	view_thread.join()
cv2.destroyAllWindows()

