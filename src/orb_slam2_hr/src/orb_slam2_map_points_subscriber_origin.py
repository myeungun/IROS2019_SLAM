#!/usr/bin/env python

##### DEFINITIONS
import sys
import rospy
import roslib
import tf2_sensor_msgs
import numpy as np
from tf2_ros import TransformStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from tf import transformations

from geometry_msgs.msg import Point, PoseStamped, Polygon
from tf import transformations # rotation_matrix(), concatenate_matrices()

import rviz_tools_py as rviz_tools

from cv_bridge import CvBridge, CvBridgeError
######

class MAP_POINTS_FROM_ORBSLAM2:

	def __init__(self):
                self.mapPoints_sub = rospy.Subscriber('/orb_slam2_zed_stereo/map_points', PointCloud2, self.PlotMapPoints)
                self.keyPoints_sub = rospy.Subscriber('/orb_slam2_zed_stereo/key_points', PointCloud2, self.KeyPointsCallback)
                self.markers_pub = rviz_tools.RvizMarkers('/map', 'visualization_marker')
                self.zed_pose_sub = rospy.Subscriber('/zed/pose', PoseStamped, self.StorePose)
                self.pose = np.zeros(3)
                self.orientation = np.zeros(4)
                self.pose_updated = False
                self.pose_update_limit = 200
                self.pose_update_limit_init = 0
                self.previous_pose = Point(0,0,0)

                self.bridge = CvBridge()
                
                self.num_sub_pose = 0

        def KeyPointsCallback(self, msg):
            if self.pose_updated == True:
                    transform_translated_xyz_0 = TransformStamped()
                    transform_translated_xyz_0.transform.translation.x = 0
                    transform_translated_xyz_0.transform.translation.y = 0
                    transform_translated_xyz_0.transform.translation.z = 0
                    transform_translated_xyz_0.transform.rotation.w = 1
                    cloud_out = tf2_sensor_msgs.do_transform_cloud(msg, transform_translated_xyz_0)
                    new_points = np.asarray(list(point_cloud2.read_points(cloud_out)), dtype=np.float32)
                    print ("KeyPoints: ", len(new_points))


        def StorePose(self, msg):
            self.pose[0] = msg.pose.position.x
            self.pose[1] = msg.pose.position.y
            self.pose[2] = msg.pose.position.z
            self.orientation[0] = msg.pose.orientation.x
            self.orientation[1] = msg.pose.orientation.y
            self.orientation[2] = msg.pose.orientation.z
            self.orientation[3] = msg.pose.orientation.w

            polygon = Polygon()
            polygon_radius = 0.1
            polygon.points.append( Point(self.pose[0]-polygon_radius, self.pose[1]+polygon_radius, self.pose[2]) )
            polygon.points.append( Point(self.pose[0]-polygon_radius, self.pose[1]-polygon_radius, self.pose[2]) )
            polygon.points.append( Point(self.pose[0]+polygon_radius, self.pose[1]-polygon_radius, self.pose[2]) )
            polygon.points.append( Point(self.pose[0]+polygon_radius, self.pose[1]+polygon_radius, self.pose[2]) )
            
            if self.pose_updated == False:
                self.pose_updated = True
                self.markers_pub.publishPolygon(polygon, 'blue', 0.02, 0)
                self.markers_pub.publishSphere(Point(self.pose[0], self.pose[1], self.pose[2]),'green', 0.025, 0)
                self.previous_pose = Point(self.pose[0], self.pose[1], self.pose[2])

            self.num_sub_pose += 1
            if self.num_sub_pose > self.pose_update_limit:
                self.markers_pub.publishPolygon(polygon, 'blue', 0.02, 0)
                self.markers_pub.publishLine(self.previous_pose, Point(self.pose[0], self.pose[1], self.pose[2]), 'green', 0.01, 0)
                self.num_sub_pose = 0
                self.previous_pose = Point(self.pose[0], self.pose[1], self.pose[2])


        def PlotMapPoints(self, msg):
                if self.pose_updated == True:
                    transform_translated_xyz_0 = TransformStamped()
                    transform_translated_xyz_0.transform.translation.x = 0
                    transform_translated_xyz_0.transform.translation.y = 0
                    transform_translated_xyz_0.transform.translation.z = 0
                    transform_translated_xyz_0.transform.rotation.w = 1
                    cloud_out = tf2_sensor_msgs.do_transform_cloud(msg, transform_translated_xyz_0)
                    new_points = np.asarray(list(point_cloud2.read_points(cloud_out)), dtype=np.float32)

                    points = []
                    print len(new_points)
                    for i in range(len(new_points)):
                        points.append(Point(new_points[i][0], new_points[i][1], new_points[i][2]))
                    diameter = 0.01

                    rgb_color = [0,255,0]                
                    self.markers_pub.publishSpheres(points, rgb_color, diameter, 0)
                    
        def unsubscribe_MAPPOINTS(self):
		self.mapPoints_sub.unregister()
######


def main(args):
	rospy.init_node('Map_Points_ORB_SMAL2_HR')
	
	ic = MAP_POINTS_FROM_ORBSLAM2()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	###
	print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)
