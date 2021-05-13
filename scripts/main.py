#!/usr/bin/env python3
import rospy
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header, Int32MultiArray, Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
from ros_3d_bb.msg import BoundingBox3D, BoundingBox3DArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
import message_filters
import threading
import queue
import time
import pyrealsense2 as rs2
import cProfile
from timer import Timer
import jax.numpy as jnp


# Send log to output?
VERBOSE = False
# Send extra log to output?
DEBUG = False
# Publish the bounding box median point in 3D?
PUBLISH_BB = True
# Publish the color and depth images?
PUBLISH_IMG = False
# Modify the color image to show samples, bb corners and the calculated point in pixels?
MODIFY_COLOR_IMAGE = False
# Modify the depth image to normalize it?
MODIFY_DEPTH_IMAGE = False
# Measure performance?
TIMING = True
# Measure performance in more detail?
TIMING_DETAILED = False
# Use the optimized version? "None" to use both
USE_OPTIMIZED = True
# Simulation?
SIMULATION = True


class Ros_3d_bb:
    """A class for converting 2D bounding boxes to 3D

    See the __init__() method's documentation for information about the various parameters.

    Attributes
    ----------
    Here are some of the key attributes which are also published (if so specified):

    raw_image : OpenCV image (color)
        The unchanged color image
    color_image: OpenCV image (color)
        Color image with extra information about the bounding boxes
    depth_image: OpenCV image (depth, 16UC1)
        The depth information from the camera
    bounding_boxes_3d : dict
        A dictionary of 3D bounding boxes
        The keys are the 2D bounding box corners and
        the values are the corresponding 3D bounding box data. 

    Methods
    -------
    There are no manually called methods, callbacks are used when messages are received.

    Subscribed topics
    -----------------
    Different topics are subscribed to depending on whether the camera is being simulated.
    All the topics are listed as the __init__() method's parameters.
    """    
    def __init__(
        self, 
        camera_info_topic_sim="/camera/aligned_depth_to_color/camera_info", 
        color_image_topic_sim="/camera/color/image_raw", 
        depth_image_topic_sim="/camera/aligned_depth_to_color/image_raw", 
        camera_info_topic_real="/realsense/color/camera_info", 
        color_image_topic_real="/realsense/color/image_raw", 
        depth_image_topic_real="/realsense/depth/image_rect_raw", 
        bb_topic="/yolo_bounding_box", 
        depth_scale=0.001, 
        frame_id="realsense_mount", 
        raw_image_out_topic="/ros_3d_bb/raw", 
        color_image_out_topic="/ros_3d_bb/color", 
        depth_image_out_topic="/ros_3d_bb/depth", 
        bb_3d_out_topic="/ros_3d_bb/bb_3d", 
        max_time_difference=0.1, 
        queue_size=10, 
        ):
        """Initializes the Ros_3d_bb node

        Parameters
        ----------
        camera_info_topic_sim : str, optional
            CameraInfo topic of the simulated camera, by default "/camera/aligned_depth_to_color/camera_info"
            For camera_info, use either "aligned_depth_to_color/camera_info" or "color/camera_info"!
        color_image_topic_sim : str, optional
            Image topic of the simulated camera, by default "/camera/color/image_raw"
        depth_image_topic_sim : str, optional
            Image topic of the simulated camera, by default "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic_real : str, optional
            CameraInfo topic of the real camera, by default "/realsense/color/camera_info"
            For camera_info, use either "aligned_depth_to_color/camera_info" or "color/camera_info"!
            Also, see the documentation about launching the RealSense ROS wrapper. 
        color_image_topic_real : str, optional
            Image topic of the real camera, by default "/realsense/color/image_raw"
        depth_image_topic_real : str, optional
            Image topic of the real camera, by default "/realsense/depth/image_rect_raw"
        bb_topic : str, optional
            2D bounding box topic, by default "/yolo_bounding_box"
            The messages have to be of type Int32MultiArray, where each bounding box
            is described by 4 consecutive corner coordinates (x_1, y_1, x_2, y_2).
            In the case of multiple bounding boxes, the array is extended (to a multiple of 4).
        depth_scale : float, optional
            The scale of one bit in meters, by default 0.001 (m, that is 1 mm)
            This code is tested with the Intel® RealSense™ Depth Camera D435i.
            For other cameras the depth scaling may or may not need to be changed.
        frame_id : str, optional
            The coordinate frame of the camera, by default "realsense_mount"
            See ROS tf for more information.
        raw_image_out_topic : str, optional
            The unmodified color image output topic, by default "/ros_3d_bb/raw"
        color_image_out_topic : str, optional
            The modified color image output topic, by default "/ros_3d_bb/color"
            The modifications can include the corners of the 2D bounding boxes,
            their 3D centers converted back to 2D and the samples used in the calculation.
        depth_image_out_topic : str, optional
            The depth image output topic, by default "/ros_3d_bb/depth"
            The output image can be normalized for better visual understanding.
        bb_3d_out_topic : str, optional
            The 3D bounding box output topic, by default "/ros_3d_bb/bb_3d"
            The published messages are of type BoundingBox3DArray.
        max_time_difference : float, optional
            Maximum time between the images and the 2D bounding box message, by default 0.1 (s)
            See message_filters.ApproximateTimeSynchronizer for further information.
        queue_size : int, optional
            Maximum queue size for the synchronization, by default 10
            See message_filters.ApproximateTimeSynchronizer for further information.
        """                
        # Variables to hold both the current color and depth image + bounding box etc
        self.raw_image = 0
        self.color_image = 0
        self.depth_image = 0
        self.corner_top_left = 0
        self.corner_bottom_right = 0
        self.bounding_boxes_3d = {}
        # Using the RealSense D435i, this value should be 0.001 (1 mm scale).
        self.depth_scale = depth_scale
        # The ROS coordinate frame of the camera
        self.frame_id = frame_id

        if TIMING:
            self.timers = []

        # Subscribed topics
        if not SIMULATION:
            self.camera_info_topic = camera_info_topic_sim
            self.color_image_topic = color_image_topic_sim
            self.depth_image_topic = depth_image_topic_sim
        else:
            # For camera_info, use either "aligned_depth_to_color/camera_info" or "color/camera_info"
            self.camera_info_topic = camera_info_topic_real
            self.color_image_topic = color_image_topic_real
            self.depth_image_topic = depth_image_topic_real

        # This is needed for both simulation and real inputs
        self.bb_topic = bb_topic

        # Published topics
        if PUBLISH_IMG:
            self.raw_image_out_topic = raw_image_out_topic
            self.color_image_out_topic = color_image_out_topic
            self.depth_image_out_topic = depth_image_out_topic
        if PUBLISH_BB:
            self.bb_3d_out_topic = bb_3d_out_topic

        # For matching different topics' messages based on timestamps:
        self.max_time_difference = max_time_difference
        self.queue_size = queue_size

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
        if PUBLISH_IMG:
            self.raw_out_pub = rospy.Publisher(
                self.raw_image_out_topic, Image, queue_size=1
            )
            self.color_out_pub = rospy.Publisher(
                self.color_image_out_topic, Image, queue_size=1
            )
            self.depth_out_pub = rospy.Publisher(
                self.depth_image_out_topic, Image, queue_size=1
            )
        if PUBLISH_BB:
            self.bb_3d_out_pub = rospy.Publisher(
                self.bb_3d_out_topic, BoundingBox3DArray, queue_size=1
            )

    def start_subscribing(self):
        """Creating the specified ROS subscribers."""

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

        # For the bounding boxes:
        bb_sub = message_filters.Subscriber(
            self.bb_topic, Int32MultiArray, queue_size=1
        )
        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.bb_topic)

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
        """Syncs the camera intrinsics.

        Copies the information from a CameraInfo ROS message
        to local rs2.instrinsics().
        This enables the calculation of x and y coordinates.

        Parameters
        ----------
        camera_info : CameraInfo (sensor_msgs.msg)
            The camera's instrinsics and extrinsics
        """

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
                rospy.loginfo("Camera instrinsics:\n" + str(self.intrinsics))

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
        """Converts the ROS Image to an OpenCV image.

        Parameters
        ----------
        color_image : Image (sensor_msgs.msg)
            A ROS Image message
        """

        try:
            self.color_image = self.bridge.imgmsg_to_cv2(color_image, "bgr8")
            self.raw_image = self.color_image.copy()
        except CvBridgeError as err:
            rospy.logerr(err)

    def depth_callback(self, depth_image, simulation=False):
        """Converts a ROS Image to an OpenCV image (depth).

        Parameters
        ----------
        depth_image : Image (sensor_msgs.msg)
            A ROS Image message containing the depth information
        simulation : bool, optional
            Whether using a real camera or a Gazebo simulation,
            by default False
        """
        try:

            if not simulation:
                # Real camera:
                self.depth_image = self.bridge.imgmsg_to_cv2(
                    depth_image, depth_image.encoding
                )
            else:
                # Simulated camera:
                self.depth_image = self.bridge.imgmsg_to_cv2(
                    depth_image, depth_image.encoding
                )
                self.depth_image *= 1000
                self.depth_image = self.depth_image.astype(np.uint16)
        except CvBridgeError as err:
            rospy.logerr(err)

    def pixel_to_point(self, x, y):
        if self.intrinsics:
            # Important! Scaling by a factor of 0.001 so the results are in meters, not mm!
            depth = self.depth_image[y, x] * self.depth_scale
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
        """Calculates the 3D bounding box coordinates and dimensions.

        This is the core method of the module.
        It is intentionally long for performance reasons,
        also enabling to use two different versions:
        one for visualization and one for optimized performance.

        Returns
        -------
        tuple
            The coordinates (x, y, z) and 
            the dimensions (size_x, size_y, size_z) of a bounding box
            in a sextuple
            If not a valid detection, the following tuple is returned:
            (0, 0, 0, 0, 0, 0)
        """

        if TIMING_DETAILED:
            timer = Timer("bb_to_coord")

        bb_width = self.corner_bottom_right[0] - self.corner_top_left[0]
        bb_height = self.corner_bottom_right[1] - self.corner_top_left[1]

        if TIMING_DETAILED:
            timer.update()

        # Note:
        # Strides can be dynamic in order to gain a slight performance advantage.
        # Right now left as-is to be compatible with the noise analysis.
        stride_x = 20
        stride_y = 20

        if TIMING_DETAILED:
            timer.update()

        # Finds the amount of samples in both directions.
        times_x = int(np.ceil(bb_width / stride_x))
        times_y = int(np.ceil(bb_height / stride_y))

        if TIMING_DETAILED:
            timer.update()

        # Offset is needed to center the samples.
        offset_x = int((bb_width - (times_x - 1) * stride_x - 1) // 2)
        offset_y = int((bb_height - (times_y - 1) * stride_y - 1) // 2)

        if TIMING_DETAILED:
            timer.stop()

        # Original version
        if not USE_OPTIMIZED:
            bb_points = np.zeros((bb_height, bb_width, 3))

            for x in range(
                    self.corner_top_left[0] + offset_x,
                    self.corner_bottom_right[0],
                    stride_x):
                for y in range(
                        self.corner_top_left[1] + offset_y,
                        self.corner_bottom_right[1],
                        stride_y):
                    point = self.pixel_to_point(x, y)
                    bb_points[
                        y - self.corner_top_left[1], x -
                        self.corner_top_left[0]
                    ] = point

                    # Display a little circle on the RGB image where each sample is located
                    if MODIFY_COLOR_IMAGE:
                        circle_color = (
                            0, 0, 255) if point[2] > 0 else (255, 0, 0)
                        cv2.circle(self.color_image, (x, y),
                                   2, circle_color, 2)

            # Only the points with non-zero depth
            filtered_points = bb_points[
                np.where(bb_points[:, :, 2] > 0)
            ]

            # Return the median of each axis
            old_medians = (
                np.nanmedian(filtered_points[:, 0]),
                np.nanmedian(filtered_points[:, 1]),
                np.nanmedian(filtered_points[:, 2]),
            )
            medians = old_medians

        # Optimized version
        # A VERY important note:
        # only use this version when the camera's distortion coefficients are 0!!!
        if USE_OPTIMIZED or USE_OPTIMIZED is None:
            if TIMING_DETAILED:
                start_times = []
                stop_times = []
                start_times.append(time.perf_counter())

            # Utilizing Numpy functions in order to get rid of a double for-loop.
            # Performing sampling with given strides in both directions.
            x_range = np.arange(
                self.corner_top_left[0] + offset_x, self.corner_bottom_right[0], stride_x)
            y_range = np.arange(
                self.corner_top_left[1] + offset_y, self.corner_bottom_right[1], stride_y)
            xx, yy = np.meshgrid(x_range, y_range)
            # Important! Scaling by a factor of 0.001 so the results are in meters, not mm!
            depths = self.depth_image[yy, xx] * self.depth_scale
            # Creating an empty array to hold the values that are going to be calculated.
            points = np.zeros((times_y, times_x, 3))

            # Note:
            # Displaying the samples on the color image is not feasible with the optimized version.
            # Therefore, if needed, consider using the unoptimized version at a slight
            # performance cost.

            if TIMING_DETAILED:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            # Calculating the points if the camera's parameters are known.
            if self.intrinsics:
                # Vectorized variants of the calculations found in the RealSense library
                # https://github.com/IntelRealSense/librealsense/blob/v2.24.0/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py#L121-L147
                points[:, :, 0] = (xx - self.intrinsics.ppx) / \
                    self.intrinsics.fx * depths
                points[:, :, 1] = (yy - self.intrinsics.ppy) / \
                    self.intrinsics.fy * depths
                points[:, :, 2] = depths

            if TIMING_DETAILED:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            # Only the points with non-zero depth (new)
            filtered_points_new = points[
                np.where(points[:, :, 2] > 0)
            ]

            if TIMING_DETAILED:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            # The following is to catch false detections.
            # Shape of "filtered_points_new":
            #   first dimension: nr of points sampled,
            #   second dimension: 3 (that is x, y and z).
            # Therefore, if the nr of samples is 0, it's not a valid detection.
            if np.shape(filtered_points_new)[0] > 0:
                new_medians = np.nanmedian(filtered_points_new, axis=0)
            else:
                # (0, 0, 0) is indicating an invalid detection
                # (0 depth signifies error as per the RealSense library).
                new_medians = (0, 0, 0)

            if TIMING_DETAILED:
                timing = time.perf_counter()
                stop_times.append(timing)
                start_times.append(timing)

            new_medians = tuple(new_medians)
            medians = new_medians

            if TIMING_DETAILED:
                # cp.disable()
                # cp.print_stats()
                timing = time.perf_counter()
                stop_times.append(timing)
                timings = np.asarray(stop_times) - np.asarray(start_times)
                rospy.loginfo("Timings: " + str(timings * 1000) + " ms")

        if USE_OPTIMIZED is None:
            rospy.loginfo("Medians:\n" +
                          "Old: " + str(old_medians) + "\n" +
                          "New: " + str(new_medians))

        # Added the scaling information for x and y (and also z but yet unimplemented).
        output_with_scale = []
        output_with_scale.extend(medians)
        # Note: this results in (0, 0, 0, 0, 0, 0) when the median depth is 0.
        # This is desired, as it allows to filter out false results later.
        output_with_scale.extend(self.get_bb_scale(
            medians[2], bb_width, bb_height))

        return tuple(output_with_scale)

    def bounding_box_callback(self, bb_multiarray):
        """Converts 2D bounding boxes to 3D.

        Extract as many 2D bounding box corner coordinates as found in the ROS message.
        Then uses the depth image to convert the 2D bounding box to 3D.

        If MODIFY_COLOR_IMAGE, draws on the image (for each bounding box):
            1) corners of the 2D bounding box,
            2) center of the 3D bounding box (converted back to 2D),
            3) if not USE_OPTIMIZED, all the samples used.

        Parameters
        ----------
        bb_multiarray : Int32MultiArray
            An array of 2D bounding box corners, 
            with each bounding box described by 4 consecutive values.
            May contain corners of zero to many bounding boxes.

        Side effects
        ------------
        For each bounding box, places the 2D bounding box corners and 3D bounding box
        data as a key-value pair into the attribute "bounding_boxes_3d".
        """

        # Note:
        # Not using custom messages for compatibility reasons.
        # Also, commons_msgs does not have a suitable array type available.

        if TIMING_DETAILED:
            timer = Timer("bb_callback")

        nr_of_bounding_boxes = len(bb_multiarray.data) // 4
        self.bounding_boxes_3d = {}

        if TIMING_DETAILED:
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

            if TIMING_DETAILED:
                timer.update()

            # Find the x, y, z and the corresponding sizes based on the 2D bounding box.
            bb_data = self.bounding_box_to_coordinates()
            self.bounding_boxes_3d[(self.corner_top_left,
                                    self.corner_bottom_right)] = bb_data

            if TIMING_DETAILED:
                timer.update()

            # Project the point with the median x, y, z back into a pixel to display it.
            # Can be disabled for performance reasons:
            if MODIFY_COLOR_IMAGE:
                self.point_to_image(bb_data)

        if MODIFY_COLOR_IMAGE:
            self.corners_to_image()

        if VERBOSE:
            rospy.loginfo("BB 3D: " + str(self.bounding_boxes_3d))

        if TIMING_DETAILED:
            timer.stop()

    def point_to_image(self, bb_data, circle_color=(0, 255, 0)):
        """Draws a 3D point on the color image.

        Converts the given 3D point back to a pixel 
        and draws it on the color image as a small circle.

        Parameters
        ----------
        bb_data : tuple
            A sextuple that contains the x, y, z 
            and the respective dimensions of a bounding box.
            Only the first three elements (x, y, z) are needed.
        circle_color : tuple, optional
            by default green (0, 255, 0) 
        """

        pixel = rs2.rs2_project_point_to_pixel(self.intrinsics, bb_data[:3])
        if DEBUG:
            rospy.loginfo(
                "BB median center xy:" +
                str(round(pixel[0])) + " " +
                str(round(pixel[1]))
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

    def corners_to_image(self, circle_color=(255, 0, 255)):
        """Draws the 2D bounding box corners on the color image as small circles.

        Parameters
        ----------
        circle_color : tuple, optional
            by default purple (255, 0, 255)
        """

        cv2.circle(
            self.color_image,
            (self.corner_top_left[0], self.corner_top_left[1]),
            5,
            circle_color,
            2,
        )
        cv2.circle(
            self.color_image,
            (self.corner_bottom_right[0],
             self.corner_bottom_right[1]),
            5,
            circle_color,
            2,
        )

    def depths_to_image(
            self, digits=3,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.75,
            text_color=(255, 0, 255),
            bb_color=(0, 255, 0)):
        """Modifies the color image to display depth information.

        Draws the 2D bounding box rectangles on the color image and 
        displays their median depths in meters.

        Parameters
        ----------
        digits : int, optional
            The number of significant digits displayed, by default 3
        font : cv2.HersheyFonts (int), optional
            by default cv2.FONT_HERSHEY_SIMPLEX (0)
        font_scale : float, optional
            by default 0.75
        text_color : tuple, optional
            by default purple (255, 0, 255)
        bb_color : tuple, optional
            Color of the 2D bounding box rectangles, by default green (0, 255, 0)
        """

        for corners, bb_data in self.bounding_boxes_3d.items():
            depth = bb_data[2]
            text_position = (
                corners[0][0], (corners[0][1] + corners[1][1]) // 2)
            text = str(round(depth, digits)) + " m"
            cv2.putText(
                self.color_image, text, text_position, font, font_scale, text_color, thickness=2
            )
            cv2.rectangle(
                self.color_image,
                corners[0],
                corners[1],
                bb_color,
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

    def publish_bb_3d(self):
        bb_array = BoundingBox3DArray()
        stamp = rospy.Time.now()
        frame_id = self.frame_id

        for bb_data in self.bounding_boxes_3d.values():
            # Only publish valid bb_data.
            # Note: the lines
            # "if not bb_data == (0, 0, 0, 0, 0, 0):" and
            # "if not bb_data[2] == 0:"
            # are functionally equal,
            # as the depth being 0 => everything else also being 0.

            # if not bb_data == (0, 0, 0, 0, 0, 0):
            if not bb_data[2] == 0:
                bb_array.boxes.append(BoundingBox3D(
                    header=Header(stamp=stamp, frame_id=frame_id),
                    center=Pose(
                        Point(
                            x=bb_data[0],
                            y=bb_data[1],
                            z=bb_data[2]),
                        Quaternion(0, 0, 0, 1)),
                    size=Vector3(
                        bb_data[3],
                        bb_data[4],
                        bb_data[5])))

        self.bb_3d_out_pub.publish(bb_array)

    def processing(self, color_image, depth_image, bb_multiarray):
        """Controls the processing flow when a pack of messages is received.

        Calls the image conversion routines, the 3D bounding box 
        coordinate and size calculation, color image modification 
        (for visualization, if enabled) and
        publishes the 3D bounding boxes (if enabled).

        Parameters
        ----------
        color_image : Image (sensor_msgs.msg)
            Color image from the camera
        depth_image : Image (sensor_msgs.msg)
            Depth image from the camera
        bb_multiarray : Int32MultiArray (std_msgs_msg)
            An array of 2D bounding boxes from a detector
            Each bounding box comprises 4 consecutive integers (the corners).
        """

        # Calculating the 3D point
        if TIMING:
            timer = Timer("ros_3d_bb")
            self.timers.append(timer)

        self.color_callback(color_image)
        self.depth_callback(depth_image, simulation=SIMULATION)

        self.bounding_box_callback(bb_multiarray)

        # Additional processing for visualization:
        # normalizing the depth image for better visual understanding.
        if MODIFY_DEPTH_IMAGE:
            depth_image = cv2.normalize(
                self.depth_image, None, 65535, 0, cv2.NORM_MINMAX)

        # If enabled, draws the bounding boxes' rectangles and median depths as text.
        if MODIFY_COLOR_IMAGE:
            self.depths_to_image()

        # Publishing
        if PUBLISH_IMG:
            self.publish_raw_image()
            self.publish_color_image(self.color_image)
            self.publish_depth_image(depth_image)

        if PUBLISH_BB:
            self.publish_bb_3d()

        if TIMING:
            timer.stop(output_file="timings_main.txt", only_total=True, nr_of_objects=len(self.bounding_boxes_3d))

            # timing_stop = time.perf_counter()
            # timing = timing_stop - timing_start
            # self.timings.append(timing)
            # rospy.loginfo("Time elapsed:" + str(timing * 1000) + " ms")

    def shutdown(self):
        """If timing the code, output the final results."""

        if TIMING:
            averages = Timer.average_times(self.timers)
            rospy.loginfo("Average timings (in ms): " + str(averages))
        else:
            rospy.loginfo("Exiting...")


def main(args):
    rospy.init_node("ros_3d_bb", disable_signals=False)
    node = Ros_3d_bb()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main(sys.argv)
