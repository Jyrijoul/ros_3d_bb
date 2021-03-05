#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
import message_filters
import threading


# Send log to output?
VERBOSE = True


class viewer:
    def __init__(self, shutdown_event):
        # An Event to coordinate the shut down of threads.
        self.shutdown_event = shutdown_event

        # Variables to hold both the current color and depth image
        self.color_image = 0
        self.depth_image = 0

        if VERBOSE:
            rospy.loginfo("Initializing the 3D bounding box viewer module...")
            rospy.loginfo("Currently only for viewing 2D images.")

        # CvBridge for converting ROS image <-> OpenCV image
        self.bridge = CvBridge()
        if VERBOSE:
            rospy.loginfo("Created the CV Bridge.")

        # Subscribed topics
        self.color_image_topic = "/ros_3d_bb/color"
        self.depth_image_topic = "ros_3d_bb/depth"

        # Subscriptions
        self.color_image_sub = message_filters.Subscriber(
            self.color_image_topic, Image, queue_size=1)
        self.depth_image_sub = message_filters.Subscriber(
            self.depth_image_topic, Image, queue_size=1)

        # For matching different topics' messages based on timestamps:
        self.max_time_difference = 0.1
        self.queue_size = 10
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_image_sub,
                self.depth_image_sub], self.queue_size, self.max_time_difference,
            allow_headerless=False)
        # Just one callback for all the topics.
        self.time_synchronizer.registerCallback(self.process_images)

    def process_images(self, ros_color_image, ros_depth_image):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(
                ros_color_image, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(
                ros_depth_image, ros_depth_image.encoding)
        except CvBridgeError as err:
            rospy.logerr(err)

    def display_images(self):
        while not self.shutdown_event.is_set():
            try:
                depth_image = cv2.normalize(self.depth_image,
                                            None, 65535, 0, cv2.NORM_MINMAX)

                # 20 * np.log10(self.depth_image / np.amax(self.depth_image)),

                cv2.imshow("Color", self.color_image)
                cv2.imshow("Depth", self.depth_image)
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


def main():
    rospy.init_node("ros_3d_bb_viewer")
    shutdown_event = threading.Event()
    module = viewer(shutdown_event)

    # Creating the displaying thread.
    display_images_thread = threading.Thread(
        name="Processing Thread", target=module.display_images)
    display_images_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        shutdown_event.set()
        rospy.loginfo("Shutting down the 3D bounding box viewer module...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
