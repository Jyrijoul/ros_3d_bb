#!/usr/bin/env python3

import rospy
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import threading
import queue
import time
import pyrealsense2.pyrealsense2 as rs2


# Send log to output?
VERBOSE = True
# Calculate depth based on bounding boxes?
calculate_bb_depth = True


class ros_3d_bb:
    def __init__(self):
        # Variables to hold both the current color and depth image + bounding box
        self.color_image = 0
        self.depth_image = 0
        self.corner_top_left = 0
        self.corner_bottom_right = 0

        # Subscribed topics
        self.camera_info_topic = "/camera/depth/camera_info"
        self.color_image_topic = "/camera/color/image_raw"
        self.depth_image_topic = "/camera/aligned_depth_to_color/image_raw"
        self.bb_topic = "/yolo_bounding_box"

        # For matching different topics' messages based on timestamps:
        self.max_time_difference = 0.1
        self.queue_size = 10

        # Camera intrinsics
        self.intrinsics = None

        if VERBOSE:
            rospy.loginfo("Initializing the 3D bounding box module...")

        # CvBridge for converting ROS image <-> OpenCV image
        self.bridge = CvBridge()
        if VERBOSE:
            rospy.loginfo("Created the CV Bridge.")

        # Start all the subscriptions and set all the callbacks
        self.start_subscribing()

    def start_subscribing(self):
        # For the camera instrinsics:
        camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo,
                                           self.camera_info_callback, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.camera_info_topic)

        # For the color image:
        color_image_sub = message_filters.Subscriber(
            self.color_image_topic, Image, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.color_image_topic)

        # For the depth image:
        depth_image_sub = message_filters.Subscriber(
            self.depth_image_topic, Image, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.depth_image_topic)

        # For the bounding box: (if selected)
        if calculate_bb_depth:
            if VERBOSE:
                rospy.loginfo(
                    "Depth calculation from bounding boxes turned on.")
            bb_sub = message_filters.Subscriber(
                self.bb_topic, Int32MultiArray, queue_size=1)
            if VERBOSE:
                rospy.loginfo("Subscribed to topic: " + self.bb_topic)
        else:
            if VERBOSE:
                rospy.loginfo(
                    "Depth calculation from bounding boxes turned off.")

        # allow_headerless = True because some of our own messages may not have a header.
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_image_sub, depth_image_sub,
                bb_sub], self.queue_size, self.max_time_difference,
            allow_headerless=True)
        # Just one callback for all the topics.
        self.time_synchronizer.registerCallback(self.processing)

    def camera_intrinsics_from_camera_info(self, camera_info):
        try:
            # https://github.com/IntelRealSense/realsense-ros/issues/1342
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = camera_info.width
            self.intrinsics.height = camera_info.height
            self.intrinsics.ppx = camera_info.K[2]
            self.intrinsics.ppy = camera_info.K[5]
            self.intrinsics.fx = camera_info.K[0]
            self.intrinsics.fy = camera_info.K[4]
            if camera_info.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif camera_info.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in camera_info.D]
        except CvBridgeError as err:
            rospy.logerr(err)

    def camera_info_callback(self, camera_info):
        if self.intrinsics:  # This is probably not the best regarding performance...
            return
        else:
            self.camera_intrinsics_from_camera_info(camera_info)

    def color_callback(self, color_image):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(color_image, "bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)

    def depth_callback(self, depth_image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                depth_image, depth_image.encoding)
        except CvBridgeError as err:
            rospy.logerr(err)

    def pixel_to_point(self, x, y):
        if self.intrinsics:
            depth = self.depth_image[y, x]
            point = rs2.rs2_deproject_pixel_to_point(
                self.intrinsics, [x, y], depth)
        else:
            # If we don't know the camera parameters,
            # return coordinates 0, 0, 0 to mark an error.
            point = [0, 0, 0]

        return point

    def bounding_box_depth(self):
        bb_width = self.corner_bottom_right[0] - self.corner_top_left[0]
        bb_height = self.corner_bottom_right[1] - self.corner_top_left[1]

        bb_depths = np.zeros((bb_height, bb_width))

        for x in range(self.corner_top_left[0], self.corner_bottom_right[0]):
            for y in range(self.corner_top_left[1], self.corner_bottom_right[1]):
                bb_depths[y - self.corner_top_left[1], x -
                          self.corner_top_left[0]] = self.pixel_to_point(x, y)[2]

        depth = np.mean(bb_depths)
        return depth

    def bounding_box_callback(self, bb_multiarray):
        # bb_multiarray may contain zero to many bounding boxes,
        # each described by 4 consecutive values.

        # Extract as many bounding box corner coordinates as found:
        nr_of_bounding_boxes = len(bb_multiarray.data) // 4
        depths = []
        for i in range(nr_of_bounding_boxes):
            self.corner_top_left = (bb_multiarray.data[0 +  4 * i], bb_multiarray.data[1 +  4 * i])
            self.corner_bottom_right = (
                bb_multiarray.data[2 +  4 * i], bb_multiarray.data[3 +  4 * i])

            depths.append(self.bounding_box_depth())
        rospy.loginfo("BB average depths:" + str(depths))

    def processing(self, color_image, depth_image, bb_multiarray):
        self.color_callback(color_image)
        self.depth_callback(depth_image)
        self.bounding_box_callback(bb_multiarray)

        depth_image = cv2.normalize(self.depth_image,
                                    None, 65535, 0, cv2.NORM_MINMAX)

        # 20 * np.log10(self.depth_image / np.amax(self.depth_image)),
        # print(np.amax(self.depth_image), np.amin(self.depth_image))

        # color = [0, 255, 0]
        # c1, c2 = self.corner_top_left, self.corner_bottom_right
        # cv2.rectangle(self.color_image, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)

        # cv2.imshow("Color", self.color_image)
        # cv2.imshow("Depth", depth_image)
        # k = cv2.waitKey(10) & 0xFF
        k = 0
        if k == ord("q"):
            # Not sure whether the next line is necessary...
            rospy.loginfo("Pressed 'q' to shut down.")
            rospy.signal_shutdown("Pressed 'q' to shut down.")


def main(args):
    rospy.init_node("ros_3d_bb")
    module = ros_3d_bb()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down the 3D bounding box module...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
