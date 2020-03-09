#!/usr/bin/env python

##### DEFINITIONS
WINDOW_NAME_Origin = 'ORB_SLAM2_DEBUG_IMAGE'
WINDOW_NAME_ZED = 'ZED IMAGE'

#### ZED Resolution
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
    CAM_HEIGHT_ORB = 396
else:
   raise Exception('Invalid CAMERA MODE')
IMG_CHANNELS = 3


import sys
import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

######

class DEBUG_IMAGE_FROM_ORBSLAM2:

	def __init__(self):
		self.bridge = CvBridge()
                # For RGB Camera
		self.debugImage_sub = rospy.Subscriber('/orb_slam2_zed_stereo/debug_image', Image, self.ORB_SLAM2_DEBUG_IMAGE_Callback)
                #self.currImage_sub = rospy.Subscriber('/orb_slam2_zed_stereo/current_image', Image, self.ORB_SLAM2_DEBUG_IMAGE_Callback)
                self.imageFromZed_sub = rospy.Subscriber('/zed/left/image_rect_color', Image, self.imageFromZEDCallback)
                #self.imageFromNetwork_sub = rospy.Subscriber('/preserve_network/left/result_only_Network', Image, self.imageFromNetworkCallback)
                # Video Output
                self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_out_network = cv2.VideoWriter(('video_out_network.avi'), self.fourcc, 4.0, (CAM_WIDTH, CAM_HEIGHT))
                self.video_out_ORB_SLAM2 = cv2.VideoWriter(('video_out_ORB_SLAM2.avi'), self.fourcc, 4.0, (CAM_WIDTH, CAM_HEIGHT_ORB))

	def ORB_SLAM2_DEBUG_IMAGE_Callback(self, msg):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imshow(WINDOW_NAME_Origin, cv_image)
		cv2.waitKey(1)
                self.video_out_ORB_SLAM2.write(cv_image)

        def imageFromNetworkCallback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                print(e)
            #cv2.imshow("IMG From Network", cv_image)
            #cv2.waitKey(1)
            self.video_out_network.write(cv_image)

        def imageFromZEDCallback(self, msg):
                try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError as e:
                        print(e)
                cv2.imshow(WINDOW_NAME_ZED, cv_image)
                cv2.waitKey(1)

	def unsubscribe_debugImageFromORBSLAM2(self):
		self.debugImage_sub.unregister()
######


def main(args):
	global print_thread_running	
        
        #Run ros node
	rospy.init_node('Debug_Image_ORB_SMAL2_HR')
	
        #Create the object of DEBUG_IMAGE_FROM_ORBSLAM2 class
	ic = DEBUG_IMAGE_FROM_ORBSLAM2()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	###
	print("Shutting down")
	cv2.destroyAllWindows()
        print("Saving videos for network results and feature extraction of ORB SLAM2.")
        ic.video_out_network.release()
        ic.video_out_ORB_SLAM2.release()

if __name__ == '__main__':
	main(sys.argv)
