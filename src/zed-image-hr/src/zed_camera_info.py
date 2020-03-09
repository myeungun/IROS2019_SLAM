#!/usr/bin/env python
##

import sys
import rospy
from sensor_msgs.msg import CameraInfo

class Camera_Info:

	def __init__(self):
		self.info_sub_left = rospy.Subscriber('/zed/left/camera_info', CameraInfo, self.leftCameraInfoCallback)

	def leftCameraInfoCallback(self, msg):
            print("Get CameraInfo")
            print("#----------------#")
            print("Width: ", msg.width)
            print("Height: ", msg.height)
            print("#----------------#")
            print("Distortion_model: ", msg.distortion_model)
            print("#----------------#")
            print("The distortion parameters, The distortion parameters")
            print("For plumb_bob, the 5 parameters are: (k1, k2, t1, t2, k3):")
            print(msg.D)
            print("#----------------#")
            print("Intrinsic camera matrix for the raw (distorted) images:")
            print("    [fx  0 cx]")
            print("K = [ 0 fy cy]")
            print("    [ 0  0  1]")
            print("Projects 3D points in the camera coordinate frame to 2D pixel")
            print("coordinates using the focal lengths (fx, fy) and principal point (cx, cy)")
            print(msg.K)
            print("#----------------#")
            print("Rectification matrix (stereo cameras only)")
            print("A rotation matrix aligning the camera coordinate system to the ideal")
            print("stereo image plane so that epipolar lines in both stereo images are parallel.")
            print(msg.R)
            print("#----------------#")
            print("Projection/camera matrix")
            print("    [fx'  0  cx' Tx]")
            print("P = [ 0  fy' cy' Ty]")
            print("    [ 0   0   1   0]")
            print("By convention, this matrix specifies the intrinsic (camera) matrix")
            print("of the processed (rectified) image. That is, the left 3x3 portion")
            print("is the normal camera intrinsic matrix for the rectified image.")
            print("It projects 3D points in the camera coordinate frame to 2D pixel")
            print("coordinates using the focal lengths (fx', fy') and principal point")
            print("(cx', cy') - these may differ from the values in K.")
            print("For a stereo pair, the fourth column [Tx Ty 0]' is related to the")
            print("position of the optical center of the second camera in the first")
            print("camera's frame. We assume Tz = 0 so both cameras are in the same")
            print("stereo image plane. The first camera always has Tx = Ty = 0. For")
            print("the right (second) camera of a horizontal stereo pair, Ty = 0 and")
            print("Tx = -fx' * B, where B is the baseline between the cameras.")
            print(msg.P)
            print("#----------------#")

def main(args):
	
        print("Node Initiation.")
	rospy.init_node('zed_Camera_Info_Subscriber_hr')
	
        print("Create the subscriber.")
	ic = Camera_Info()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)
