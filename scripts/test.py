#!/usr/bin/env python3

import rospy
# import roslib
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import threading
import queue
import time


VERBOSE = True


class ros_3d_bb:
    def __init__(self, shutdown_event):
        # An Event to coordinate the shut down of threads.
        self.shutdown_event = shutdown_event

        image_topic = "/camera/color/image_raw"
        if VERBOSE:
            rospy.loginfo("Initializing the 3D bounding-box module...")

        self.bridge = CvBridge()
        if VERBOSE:
            rospy.loginfo("Created the CV Bridge.")

        if VERBOSE:
            rospy.loginfo("Trying to subscribe to topic " +
                          image_topic + "...")
        self.image_sub = rospy.Subscriber(image_topic, Image,
                                          self.callback, queue_size=1)
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + image_topic)

    def callback(self, ros_image):
        #rospy.loginfo("Hello, ROS!")
        #cv2.imshow("Image", image_data)
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as err:
            rospy.logerr(err)

        # rospy.loginfo(np.shape(frame))
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(10) & 0xFF
        if k == ord("q"):
            self.shutdown_event.set()
            # Not sure whether the next line is necessary...
            rospy.loginfo("Pressed 'q' to shut down.")
            rospy.signal_shutdown("Pressed 'q' to shut down.")

    def processing(self):
        while not self.shutdown_event.is_set():
            print("I'm alive!")
            time.sleep(1)
        
        print("Exiting...")


def main(args):
    rospy.init_node("ros_3d_bb")
    shutdown_event = threading.Event()
    module = ros_3d_bb(shutdown_event)

    # Creating the processing thread.
    processing_thread = threading.Thread(target=module.processing)
    processing_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        shutdown_event.set()
        rospy.loginfo("Shutting down the 3D bounding-box module...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
