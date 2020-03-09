#!/usr/bin/env python

#### LIBRARIES

import sys
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from modules.ringbuff import RingBuffer
from operator import mul

import threading


##### DEFINITIONS

#### ZED RESOLUTION
ZED_CAMERA_RESOLUTION = 3
if ZED_CAMERA_RESOLUTION == 0:
    # 1. RESOLUTION_HD2K
    CAM_WIDTH = 2208
    CAM_HEIGHT = 1242
    CAM_CHANNELS = 3
elif ZED_CAMERA_RESOLUTION == 1:
    # 2. RESOLUTION_HD1280
    CAM_WIDTH = 1920
    CAM_HEIGHT = 1080
    CAM_CHANNELS = 3
elif ZED_CAMERA_RESOLUTION == 2:
    # 3. RESOLUTION_HD720
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    CAM_CHANNELS = 3
elif ZED_CAMERA_RESOLUTION == 3:
    # 4. RESOLUTION_VGA
    CAM_WIDTH = 672
    CAM_HEIGHT = 376
    CAM_CHANNELS = 3
else:
   raise Exception('Invalid CAMERA MODE')
IMG_CHANNELS = 3

### NAME of WINDOW to SHOW THE RESULTS
WINDOW_NAME_RIGHT = 'RIGHT LOW IMAGE'
WINDOW_NAME_LEFT = 'LEFT LOW IMAGE'

BUFF_SIZE = 80
IMG_SHAPE_ORIGIN = (CAM_HEIGHT, CAM_WIDTH, CAM_CHANNELS)
IMG_SIZE_ORIGIN = reduce(mul, IMG_SHAPE_ORIGIN)

IMGBUFF_ORIGIN = IMG_SIZE_ORIGIN * BUFF_SIZE

### Buffer Initialization
img_buffer_right = RingBuffer(IMGBUFF_ORIGIN, 'float32')
img_buffer_right_lock = threading.RLock()
img_buffer_left = RingBuffer(IMGBUFF_ORIGIN, 'float32')
img_buffer_left_lock = threading.RLock()
###

### LOW RESOLUTION
LOW_IMG_WIDTH = 32
LOW_IMG_HEIGHT = 24

class image_From_ZED:

	def __init__(self):
		self.bridge = CvBridge()
		self.imageRight_sub = rospy.Subscriber('/zed/right/image_rect_color', Image, self.imageRightPushBuffer)
		self.imageLeft_sub = rospy.Subscriber('/zed/left/image_rect_color', Image, self.imageLeftPushBuffer)
		self.imageRightResult_pub = rospy.Publisher('/privacy_preserve/right/low_image', Image, queue_size=10)
		self.imageLeftResult_pub = rospy.Publisher('/privacy_preserve/left/low_image', Image, queue_size=10)

	def imageRightPushBuffer(self, msg):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        image_resized = cv2.resize(cv_image, (LOW_IMG_WIDTH, LOW_IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                        image_resized = cv2.resize(image_resized, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
		except CvBridgeError as e:
			print(e)
                if img_buffer_right.nb_data/IMG_SIZE_ORIGIN < 3:
	        	with img_buffer_right_lock:
		        	img_buffer_right.push(image_resized.flatten().astype('float32'))
	
        def imageLeftPushBuffer(self, msg):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        image_resized = cv2.resize(cv_image, (LOW_IMG_WIDTH, LOW_IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                        image_resized = cv2.resize(image_resized, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_AREA)
		except CvBridgeError as e:
			print(e)
                if img_buffer_left.nb_data/IMG_SIZE_ORIGIN < 3:
		        with img_buffer_left_lock:
			        img_buffer_left.push(image_resized.flatten().astype('float32'))

	def imageRightCallback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imshow(WINDOW_NAME_RIGHT, cv_image)
		cv2.waitKey(1)
        
        def imageLeftCallback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imshow(WINDOW_NAME_LEFT, cv_image)
		cv2.waitKey(1)


	def unsubscribe_imageFromZED(self):
		self.imageRight_sub.unregister()

### Threading Functions
def imageResultPrint():
	while print_thread_running:
                if img_buffer_right.nb_data > 0:
			with img_buffer_right_lock:
				popped_right_image = img_buffer_right.get(IMG_SIZE_ORIGIN)
				img_buffer_right.pop(IMG_SIZE_ORIGIN)
                                popped_right_image = popped_right_image.reshape(IMG_SHAPE_ORIGIN).astype('uint8')
                		cv2.imshow(WINDOW_NAME_RIGHT, popped_right_image)
                		cv2.waitKey(1)


		if img_buffer_left.nb_data > 0:
			with img_buffer_left_lock:
				popped_left_image = img_buffer_left.get(IMG_SIZE_ORIGIN)
				img_buffer_left.pop(IMG_SIZE_ORIGIN)
                		popped_left_image = popped_left_image.reshape(IMG_SHAPE_ORIGIN).astype('uint8')
		                cv2.imshow(WINDOW_NAME_LEFT, popped_left_image)
		                cv2.waitKey(1)

def resultPublisher(ResPub):

	while publish_thread_running:
		if img_buffer_right.nb_data > 0 and img_buffer_left.nb_data > 0:
			with img_buffer_right_lock:
				popped_right_image = img_buffer_right.get(IMG_SIZE_ORIGIN)
				img_buffer_right.pop(IMG_SIZE_ORIGIN)
			with img_buffer_left_lock:
				popped_left_image = img_buffer_left.get(IMG_SIZE_ORIGIN)
				img_buffer_left.pop(IMG_SIZE_ORIGIN)
                        
                        popped_right_image = popped_right_image.reshape(IMG_SHAPE_ORIGIN).astype('uint8')
			popped_left_image = popped_left_image.reshape(IMG_SHAPE_ORIGIN).astype('uint8')
			try:
				ResPub.imageRightResult_pub.publish(ResPub.bridge.cv2_to_imgmsg(popped_right_image, 'bgr8'))
				ResPub.imageLeftResult_pub.publish(ResPub.bridge.cv2_to_imgmsg(popped_left_image, 'bgr8'))
			except CvBridgeError as e:
				print('Exception Occurs...')
				print(e)

### Node initialization
rospy.init_node('zed_image_convert_low_resolution')
ZED_Image_obj = image_From_ZED()

### Threading start
print_thread_running = False
publish_thread_running = True

if print_thread_running == True:
	print_thread = threading.Thread(target=imageResultPrint)
	print_thread.start()
	print('Start print_thread...')
if publish_thread_running == True:
	publish_thread = threading.Thread(target=resultPublisher, args = (ZED_Image_obj,))
	publish_thread.start()
	print('Start publish_thread')
######

### Run nodes
try:
	rospy.spin()
except KeyboardInterrupt:
	print("Shutting down")

### Procedures for termination threads
print("Shutting down")
if print_thread_running == True:
	print_thread_running = False
	print_thread.join()
if publish_thread_running == True:
	publish_thread_running = False
	publish_thread.join()
cv2.destroyAllWindows()
