#!/usr/bin/env python3

import rospy
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray, Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import threading
import queue
import time
import pyrealsense2 as rs2
import cProfile
from timer import Timer
import jax.numpy as jnp


"""
This code is tested with the Intel® RealSense™ Depth Camera D435i.
For other cameras the depth scaling may or may not need to be changed.
If needed, change the following line:
"depths = self.depth_image[yy, xx]" to
"depths = self.depth_image[yy, xx] * depth_scale", 
where "depth_scale" is the output of rs2's "get_depth_scale" function.
"""


# Send log to output?
VERBOSE = True
# Calculate depth based on bounding boxes?
calculate_bb_depth = True
# Publish the bounding box mean point in 3D-coordinate space?
publishing_bb_point = True
# Modify the color image to show samples, bb corners and the calculated point?
modify_color_image = False
# Performance measurement
measure_performance = True
# Performance measurement in more detail
measure_detailed_performance = False
# Use the optimized version? "None" to use both
use_optimized = True


class Ros_3d_bb:
    def __init__(self):
        # Variables to hold both the current color and depth image + bounding box etc
        self.raw_image = 0
        self.color_image = 0
        self.depth_image = 0
        self.corner_top_left = 0
        self.corner_bottom_right = 0
        self.depths = {}
        self.coordinates = (0, 0, 0)

        if measure_performance:
            self.timings = []

        # Subscribed topics
        # For camera_info, use either "aligned_depth_to_color/camera_info" or "color/camera_info"
        self.camera_info_topic = "/camera/aligned_depth_to_color/camera_info"
        # self.camera_info_topic = "/camera/color/camera_info"
        self.color_image_topic = "/camera/color/image_raw"
        self.depth_image_topic = "/camera/aligned_depth_to_color/image_raw"
        """For simulation use the following:"""
        # For camera_info, use either "aligned_depth_to_color/camera_info" or "color/camera_info"
        self.camera_info_topic = "/realsense/color/camera_info"
        self.color_image_topic = "/realsense/color/image_raw"
        self.depth_image_topic = "/realsense/depth/image_rect_raw"
        self.bb_topic = "/yolo_bounding_box"

        # Published topics
        self.raw_image_out_topic = "/ros_3d_bb/raw"
        self.color_image_out_topic = "/ros_3d_bb/color"
        self.depth_image_out_topic = "/ros_3d_bb/depth"
        if publishing_bb_point:
            self.bb_point_out_topic = "/ros_3d_bb/point"

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

        # Publishers
        self.raw_out_pub = rospy.Publisher(
            self.raw_image_out_topic, Image, queue_size=1
        )
        self.color_out_pub = rospy.Publisher(
            self.color_image_out_topic, Image, queue_size=1
        )
        self.depth_out_pub = rospy.Publisher(
            self.depth_image_out_topic, Image, queue_size=1
        )
        if publishing_bb_point:
            self.bb_point_out_pub = rospy.Publisher(
                self.bb_point_out_topic, Int32MultiArray, queue_size=1
            )

    def start_subscribing(self):
        # For the camera instrinsics:
        camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1
        )
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.camera_info_topic)

        # For the color image:
        color_image_sub = message_filters.Subscriber(
            self.color_image_topic, Image, queue_size=1
        )
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.color_image_topic)

        # For the depth image:
        depth_image_sub = message_filters.Subscriber(
            self.depth_image_topic, Image, queue_size=1
        )
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.depth_image_topic)

        # For the bounding box: (if selected)
        if calculate_bb_depth:
            if VERBOSE:
                rospy.loginfo(
                    "Depth calculation from bounding boxes turned on.")
            bb_sub = message_filters.Subscriber(
                self.bb_topic, Int32MultiArray, queue_size=1
            )
            if VERBOSE:
                rospy.loginfo("Subscribed to topic: " + self.bb_topic)
        else:
            if VERBOSE:
                rospy.loginfo(
                    "Depth calculation from bounding boxes turned off.")

        # allow_headerless = True because some of our own messages may not have a header.
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_image_sub, depth_image_sub, bb_sub],
            self.queue_size,
            self.max_time_difference,
            allow_headerless=True,
        )
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
            if camera_info.distortion_model == "plumb_bob":
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif camera_info.distortion_model == "equidistant":
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in camera_info.D]
            if VERBOSE:
                print("Camera instrinsics:")
                print(self.intrinsics)

                """
                With the 435i it should be something like:
                640x480  p[319.199 242.08]  f[614.686 614.842]  Brown Conrady [0 0 0 0 0]
                """
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
            self.raw_image = self.color_image.copy()
        except CvBridgeError as err:
            rospy.logerr(err)

    def depth_callback(self, depth_image):
        try:
            """Real camera:"""
            # self.depth_image = self.bridge.imgmsg_to_cv2(
                # depth_image, depth_image.encoding
            # )

            """Simulation:"""
            self.depth_image = self.bridge.imgmsg_to_cv2(
                depth_image, depth_image.encoding
            )
            self.depth_image *= 1000
            self.depth_image = self.depth_image.astype(np.uint16)
            print(np.shape(self.depth_image))
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

    def get_bb_scale(self, median_depth, bb_width, bb_height, bb_depth=0):
        x_scale = bb_width / self.intrinsics.fx * median_depth
        y_scale = bb_height / self.intrinsics.fy * median_depth
        # TODO: Figure out how to compute the scale in the z-direction (if needed at all!).
        z_scale = 0
        return (x_scale, y_scale, z_scale)

    def bounding_box_to_coordinates(self):
        if measure_detailed_performance:
            timer = Timer("bb_to_coord")

        bb_width = self.corner_bottom_right[0] - self.corner_top_left[0]
        bb_height = self.corner_bottom_right[1] - self.corner_top_left[1]

        if measure_detailed_performance:
            timer.update()

        stride_x = 20
        stride_y = 20

        if measure_detailed_performance:
            timer.update()

        times_x = int(np.ceil(bb_width / stride_x))
        times_y = int(np.ceil(bb_height / stride_y))

        if measure_detailed_performance:
            timer.update()

        # Offset is needed to center the samples.
        offset_x = int((bb_width - (times_x - 1) * stride_x - 1) // 2)
        offset_y = int((bb_height - (times_y - 1) * stride_y - 1) // 2)

        if measure_detailed_performance:
            timer.stop()

        # Original version
        if not use_optimized:
            bb_points = np.zeros((bb_height, bb_width, 3))

            # if VERBOSE:
            #     print(offset_x, offset_y)
            #     print("bb_points shape:", np.shape(bb_points))
            for x in range(
                self.corner_top_left[0] +
                    offset_x, self.corner_bottom_right[0], stride_x
            ):
                for y in range(
                    self.corner_top_left[1] + offset_y,
                    self.corner_bottom_right[1],
                    stride_y,
                ):
                    point = self.pixel_to_point(x, y)
                    bb_points[
                        y - self.corner_top_left[1], x -
                        self.corner_top_left[0]
                    ] = point
                    # Display a little circle on the RGB image where each sample is located
                    circle_color = (0, 0, 255) if point[2] > 0 else (255, 0, 0)
                    cv2.circle(self.color_image, (x, y), 2, circle_color, 2)

            # Only the points with non-zero depth
            filtered_points = bb_points[
                np.where(bb_points[:, :, 2] > 0)
            ]  # Horrible syntax, I know!

            # Return the median of each axis
            old_medians = (
                np.nanmedian(filtered_points[:, 0]),
                np.nanmedian(filtered_points[:, 1]),
                np.nanmedian(filtered_points[:, 2]),
            )
            medians = old_medians

        # Optimized version
        # A VERY important note:
        # only use this when the camera's distortion coefficients are 0!!!
        if use_optimized or use_optimized is None:
            if measure_detailed_performance:
                start_times = []
                stop_times = []
                start_times.append(time.perf_counter())
            x_range = np.arange(
                self.corner_top_left[0] + offset_x, self.corner_bottom_right[0], stride_x)
            y_range = np.arange(
                self.corner_top_left[1] + offset_y, self.corner_bottom_right[1], stride_y)
            xx, yy = np.meshgrid(x_range, y_range)
            depths = self.depth_image[yy, xx]
            points = np.zeros((times_y, times_x, 3))

            if measure_detailed_performance:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            # We can only calculate the points if we know the camera's parameters.
            if self.intrinsics:
                # https://github.com/IntelRealSense/librealsense/blob/v2.24.0/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py#L121-L147
                points[:, :, 0] = (xx - self.intrinsics.ppx) / \
                    self.intrinsics.fx * depths
                points[:, :, 1] = (yy - self.intrinsics.ppy) / \
                    self.intrinsics.fy * depths
                points[:, :, 2] = depths

            if measure_detailed_performance:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            # Only the points with non-zero depth (new)
            filtered_points_new = points[
                np.where(points[:, :, 2] > 0)
            ]

            if measure_detailed_performance:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            # print(np.shape(filtered_points_new))
            if np.shape(filtered_points_new)[0] > 0:
                new_medians = np.nanmedian(filtered_points_new, axis=0)
            else:
                new_medians = (0, 0, 0)

            if measure_detailed_performance:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            new_medians = tuple(new_medians)
            medians = new_medians

            if measure_detailed_performance:
                # cp.disable()
                # cp.print_stats()
                timing = time.perf_counter()
                stop_times.append(timing)
                timings = np.asarray(stop_times) - np.asarray(start_times)
                print("Timings:", timings * 1000, " ms")

        if use_optimized is None:
            print("Medians:")
            print("Old:", old_medians)
            print("New:", new_medians)

        # Added the scaling information for x and y (and also z but yet unimplemented).
        output_with_scale = []
        output_with_scale.extend(medians)
        output_with_scale.extend(self.get_bb_scale(
            medians[2], bb_width, bb_height))

        return output_with_scale

    def bounding_box_callback(self, bb_multiarray):
        # img_height, img_width = np.shape(self.color_image)[:2]
        # if VERBOSE:
        # print(img_height, img_width)

        # bb_multiarray may contain zero to many bounding boxes,
        # each described by 4 consecutive values.

        # Extract as many bounding box corner coordinates as found:

        if measure_detailed_performance:
            timer = Timer("bb_callback")

        nr_of_bounding_boxes = len(bb_multiarray.data) // 4
        self.points = {}

        if measure_detailed_performance:
            timer.update()

        for i in range(nr_of_bounding_boxes):
            self.corner_top_left = (
                bb_multiarray.data[0 + 4 * i],
                bb_multiarray.data[1 + 4 * i],
            )
            self.corner_bottom_right = (
                bb_multiarray.data[2 + 4 * i],
                bb_multiarray.data[3 + 4 * i],
            )

            if measure_detailed_performance:
                timer.update()

            # Find the x, y and z (in mm) based on the bounding box
            point = self.bounding_box_to_coordinates()
            self.points[(self.corner_top_left,
                         self.corner_bottom_right)] = point

            if measure_detailed_performance:
                timer.update()

            # Project the point with the mean x, y, z back into a pixel to display it
            # Can be disables for performance reasons:
            if modify_color_image:
                circle_color = (0, 255, 0)
                pixel = rs2.rs2_project_point_to_pixel(self.intrinsics, point)
                if VERBOSE:
                    print(
                        "BB mean center xy:",
                        round(pixel[0]),
                        round(pixel[1]),
                    )
                cv2.circle(
                    self.color_image,
                    (
                        round(pixel[0]),
                        round(pixel[1]),
                    ),
                    5,
                    circle_color,
                    2,
                )

        if modify_color_image:
            circle_color = (255, 0, 255)
            cv2.circle(
                self.color_image,
                (self.corner_top_left[0], self.corner_top_left[1]),
                5,
                circle_color,
                2,
            )
            circle_color = (255, 0, 255)
            cv2.circle(
                self.color_image,
                (self.corner_bottom_right[0], self.corner_bottom_right[1]),
                5,
                circle_color,
                2,
            )

        if VERBOSE:
            # Only show the coordinates, not the scaling information.
            rospy.loginfo("Points: " + str(self.points))

        if measure_detailed_performance:
            timer.stop()

    def depths_to_image(self):
        # print(self.depths.keys(), "len:", len(self.depths.keys()))
        for corners, depth in self.depths.items():
            corner = (corners[0][0], (corners[0][1] + corners[1][1]) // 2)
            # print("Corners:", corners)
            text = str(round(depth, 1)) + " mm"
            cv2.putText(
                self.color_image, text, corner, 0, 0.75, [255, 0, 255], thickness=2
            )
            color = [0, 255, 0]
            cv2.rectangle(
                self.color_image,
                corners[0],
                corners[1],
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def publish_raw_image(self):
        self.raw_out_pub.publish(
            self.bridge.cv2_to_imgmsg(self.raw_image, "bgr8"))

    def publish_color_image(self, image):
        self.color_out_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

    def publish_depth_image(self, image):
        self.depth_out_pub.publish(self.bridge.cv2_to_imgmsg(image, "16UC1"))

    def publish_bb_point(self, points_1d_array):
        message = Int32MultiArray()
        message.data = points_1d_array
        self.bb_point_out_pub.publish(message)

    def processing(self, color_image, depth_image, bb_multiarray):
        self.color_callback(color_image)
        self.depth_callback(depth_image)

        # Calculating the 3D point
        if measure_performance:
            timing_start = time.perf_counter()
        self.bounding_box_callback(bb_multiarray)
        if measure_performance:
            timing_stop = time.perf_counter()
            timing = timing_stop - timing_start
            self.timings.append(timing)
            rospy.loginfo("Time elapsed:" + str(timing * 1000) + " ms")

        # Additional processing for visualization
        depth_image = cv2.normalize(
            self.depth_image, None, 65535, 0, cv2.NORM_MINMAX)
        self.depths_to_image()

        # Publishing
        self.publish_raw_image()
        self.publish_color_image(self.color_image)
        self.publish_depth_image(depth_image)

        if publishing_bb_point:
            points = []
            for point in self.points.values():
                # Only publish valid points.
                if not point == (0, 0, 0, 0, 0, 0):
                    points.extend(list(map(round, point)))
            self.publish_bb_point(points)

    def shutdown(self):
        if len(self.timings) != 0:
            average = np.average(self.timings)
            rospy.loginfo("Average performance: " +
                          str(average * 1000) + " ms")
        else:
            rospy.loginfo("No data to calculate the average performance!")


def main(args):
    rospy.init_node("ros_3d_bb", disable_signals=False)
    module = Ros_3d_bb()
    rospy.on_shutdown(module.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)
