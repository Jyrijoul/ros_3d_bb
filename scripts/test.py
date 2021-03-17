#!/usr/bin/env python3

import rospy
# import roslib
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image, CameraInfo
import threading
import queue
import time
import pyrealsense2.pyrealsense2 as rs2


# Send log to output?
VERBOSE = True
# Calculate depth based on bounding boxes?
calculate_bb_depth = True


class Ros_3d_bb:
    def __init__(self, shutdown_event):
        # An Event to coordinate the shut down of threads.
        self.shutdown_event = shutdown_event

        # Variables to hold both the current color and depth image
        self.color_image = 0
        self.depth_image = 0

        # Camera intrinsics
        self.intrinsics = None

        if VERBOSE:
            rospy.loginfo("Initializing the 3D bounding box module...")

        self.bridge = CvBridge()
        if VERBOSE:
            rospy.loginfo("Created the CV Bridge.")

        # For the camera instrinsics:
        camera_info_topic = "/camera/depth/camera_info"
        if VERBOSE:
            rospy.loginfo("Trying to subscribe to topic " +
                          camera_info_topic + "...")
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo,
                                                self.camera_info_callback, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + camera_info_topic)

        # For the color image:
        color_image_topic = "/camera/color/image_raw"
        if VERBOSE:
            rospy.loginfo("Trying to subscribe to topic " +
                          color_image_topic + "...")
        self.color_image_sub = rospy.Subscriber(color_image_topic, Image,
                                                self.color_callback, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + color_image_topic)

        # For the depth image:
        # depth_image_topic = "/camera/depth/image_rect_raw"
        depth_image_topic = "/camera/aligned_depth_to_color/image_raw"
        if VERBOSE:
            rospy.loginfo("Trying to subscribe to topic " +
                          depth_image_topic + "...")
        self.depth_image_sub = rospy.Subscriber(depth_image_topic, Image,
                                                self.depth_callback, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + depth_image_topic)

        # For the bounding box: (if selected)
        if calculate_bb_depth:
            bb_topic = "/yolo_bounding_box"
            if VERBOSE:
                rospy.loginfo("Depth calculation from bounding boxes turned on.")
                rospy.loginfo("Trying to subscribe to topic " +
                              bb_topic + "...")
            self.bb_sub = rospy.Subscriber(bb_topic, Int32MultiArray,
                                        self.bounding_box_callback, queue_size=1)
            if VERBOSE:
                rospy.loginfo("Subscribed to topic: " + bb_topic)
        else:
            if VERBOSE:
                rospy.loginfo("Depth calculation from bounding boxes turned off.")

    def camera_info_callback(self, camera_info):
        try:
            # https://github.com/IntelRealSense/realsense-ros/issues/1342
            if self.intrinsics:  # This is probably not the best regarding performance...
                return
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
            return  # This is not working?

    def color_callback(self, ros_image):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)
            return  # This is not working?

        # rospy.loginfo(np.shape(self.color_image))

    def depth_callback(self, ros_image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                ros_image, ros_image.encoding)
        except CvBridgeError as err:
            rospy.logerr(err)
            return  # This is not working?

        # rospy.loginfo(np.shape(self.depth_image))

    def pixel_to_point(self, x, y):
        if self.intrinsics:
            depth = self.depth_image[y, x]
            point = rs2.rs2_deproject_pixel_to_point(
                self.intrinsics, [x, y], depth)
        else:
            point = [0, 0, 0]

        return point

    def bounding_box_callback(self, bb_multiarray):
        # Extracting the corners.
        c1 = (bb_multiarray.data[0], bb_multiarray.data[1])
        c2 = (bb_multiarray.data[2], bb_multiarray.data[3])

        bb_width = c2[0] - c1[0]
        bb_height = c2[1] - c1[1]

        bb_depths = np.zeros((bb_height, bb_width))

        for x in range(c1[0], c2[0]):
            for y in range(c1[1], c2[1]):
                bb_depths[y - c1[1], x - c1[0]] = self.pixel_to_point(x, y)[2]

        avg_depth = np.average(bb_depths)
        rospy.loginfo("BB average depth:" + str(avg_depth))

    def processing(self):
        while not self.shutdown_event.is_set():
            # No need to run as fast as possible (max 60 fps).
            time.sleep(0.0167)
            try:
                depth_image = cv2.normalize(self.depth_image,
                                            None, 65535, 0, cv2.NORM_MINMAX)

                # 20 * np.log10(self.depth_image / np.amax(self.depth_image)),

                cv2.imshow("Color", self.color_image)
                cv2.imshow("Depth", depth_image)
                # print(np.amax(self.depth_image), np.amin(self.depth_image))
                k = cv2.waitKey(10) & 0xFF
                if k == ord("q"):
                    self.shutdown_event.set()
                    # Not sure whether the next line is necessary...
                    rospy.loginfo("Pressed 'q' to shut down.")
                    rospy.signal_shutdown("Pressed 'q' to shut down.")
                
            except KeyboardInterrupt:
                break

        print("Exiting...")


def main(args):
    rospy.init_node("ros_3d_bb")
    shutdown_event = threading.Event()
    module = Ros_3d_bb(shutdown_event)

    # Creating the processing thread.
    processing_thread = threading.Thread(
        name="Processing Thread", target=module.processing)
    processing_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        shutdown_event.set()
        rospy.loginfo("Shutting down the 3D bounding box module...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
