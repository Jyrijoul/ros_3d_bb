#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
import message_filters
import threading
import datetime


# Send log to output?
VERBOSE = True
# Turn on measurement taking system?
taking_measurements = False
# If taking measurements, specify the measurement output file name base:
measurements_file_base = "measurements"


class Viewer:
    def __init__(self, shutdown_event):
        # An Event to coordinate the shut down of threads.
        self.shutdown_event = shutdown_event

        # Variables to hold both the current color, depth image and points
        self.raw_image = 0
        self.color_image = 0
        self.depth_image = 0
        self.points = []

        if VERBOSE:
            rospy.loginfo("Initializing the 3D bounding box viewer module...")
            rospy.loginfo("Currently only for viewing 2D images.")

        # CvBridge for converting ROS image <-> OpenCV image
        self.bridge = CvBridge()
        if VERBOSE:
            rospy.loginfo("Created the CV Bridge.")

        # Subscribed topics
        self.raw_image_topic = "/ros_3d_bb/raw"
        self.color_image_topic = "/ros_3d_bb/color"
        self.depth_image_topic = "/ros_3d_bb/depth"
        # if taking_measurements:
        self.bb_point_topic = "/ros_3d_bb/point"

        # Subscriptions
        self.raw_image_sub = message_filters.Subscriber(
            self.raw_image_topic, Image, queue_size=1
        )
        self.color_image_sub = message_filters.Subscriber(
            self.color_image_topic, Image, queue_size=1
        )
        self.depth_image_sub = message_filters.Subscriber(
            self.depth_image_topic, Image, queue_size=1
        )
        # if taking_measurements:
        # self.bb_point_sub = message_filters.Subscriber(
        #     self.bb_point_topic, Int32MultiArray, queue_size=1
        # )

        self.bb_point_alt_sub = rospy.Subscriber(
            self.bb_point_topic, Int32MultiArray, self.bb_point_alt_callback, queue_size=1
        )

        # For matching different topics' messages based on timestamps:
        self.max_time_difference = 0.1
        self.queue_size = 10
        # if not taking_measurements:
        #     self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
        #         [self.color_image_sub, self.depth_image_sub],
        #         self.queue_size,
        #         self.max_time_difference,
        #         allow_headerless=True,
        #     )
        # else:
        #     self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
        #         [self.color_image_sub, self.depth_image_sub, self.bb_point_sub],
        #         self.queue_size,
        #         self.max_time_difference,
        #         allow_headerless=True,
        #     )
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub, self.raw_image_sub],
            self.queue_size,
            self.max_time_difference,
            allow_headerless=True,
        )
        # Just one callback for all the topics.
        self.time_synchronizer.registerCallback(self.process_images)

    def bb_point_alt_callback(self, bb_point_multiarray):
        # print(bb_point_multiarray)
        self.process_bb_point_multiarray(bb_point_multiarray)

    def process_images(self, ros_color_image, ros_depth_image, raw_image, bb_point_multiarray=0):
        try:
            # print("Start")
            self.raw_image = self.bridge.imgmsg_to_cv2(raw_image, "bgr8")
            self.color_image = self.bridge.imgmsg_to_cv2(ros_color_image, "bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(
                ros_depth_image, ros_depth_image.encoding
            )
            # If taking measurements, also process the coordinates.
            # if taking_measurements:
            #     #self.process_bb_point_multiarray(bb_point_multiarray)
            #     print(bb_point_multiarray)
        except CvBridgeError as err:
            rospy.logerr(err)

    def process_bb_point_multiarray(self, bb_point_multiarray):
        # bb_point_multiarray may contain zero to many points,
        # each described by 6 consecutive values.

        # Extract as many point x, y and z coordinates as found:
        nr_of_bb_points = len(bb_point_multiarray.data) // 6
        self.points = []
        for i in range(nr_of_bb_points):
            self.points.append(
                (
                    bb_point_multiarray.data[0 + 6 * i],
                    bb_point_multiarray.data[1 + 6 * i],
                    bb_point_multiarray.data[2 + 6 * i],
                )
            )

    def display_images(self):
        cv2.namedWindow('Color', cv2.WINDOW_NORMAL)
        while not self.shutdown_event.is_set():
            try:
                depth_image = cv2.normalize(
                    self.depth_image, None, 65535, 0, cv2.NORM_MINMAX
                )

                # 20 * np.log10(self.depth_image / np.amax(self.depth_image)),
                cv2.imshow("Raw color", self.raw_image)
                cv2.imshow("Depth", self.depth_image)
                cv2.imshow("Color", self.color_image)
                # print(np.amax(self.depth_image), np.amin(self.depth_image))

                # print(self.points)

                if taking_measurements:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord("q"):
                        self.shutdown_event.set()
                        # Not sure whether the next line is necessary...
                        rospy.loginfo("Pressed 'q' to shut down.")
                        rospy.signal_shutdown("Pressed 'q' to shut down.")
                    elif k == ord("s"):
                        now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                        print(now)
                        cv2.imwrite(measurements_file_base + "_" + now + "_raw.png", self.raw_image)
                        cv2.imwrite(measurements_file_base + "_" + now + "_color.png", self.color_image)
                        cv2.imwrite(measurements_file_base + "_" + now + "_depth.png", self.depth_image)
                        with open(measurements_file_base + ".txt", "a") as f:
                            f.write(str(self.points) + ";" + now + "\n")
                else:
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
    module = Viewer(shutdown_event)

    # Creating the displaying thread.
    display_images_thread = threading.Thread(
        name="Displaying Thread", target=module.display_images
    )
    display_images_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        shutdown_event.set()
        rospy.loginfo("Shutting down the 3D bounding box viewer module...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
