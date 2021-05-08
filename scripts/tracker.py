#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
import numpy as np
from scipy.spatial import distance
import time
import rviz_util
import predictor
from timer import Timer
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped, Vector3Stamped
from ros_3d_bb.msg import BoundingBox3D, BoundingBox3DArray
import tf2_ros
import tf2_geometry_msgs


VISUALIZATION = True
VERBOSE = True
DEBUG = True
TIME = True


class BoundingBox:
    def __init__(self, x=0, y=0, z=0, size_x=0, size_y=0, size_z=0, bounding_box=None):
        """Create a bounding box with all values passed separately or using a BoundingBox3D message"""

        if not bounding_box:
            self.x = x
            self.y = y
            self.z = z
            self.size_x = size_x
            self.size_y = size_y
            self.size_z = size_z
        else:
            self.x = bounding_box.center.position.x
            self.y = bounding_box.center.position.y
            self.z = bounding_box.center.position.z
            self.size_x = bounding_box.size.x
            self.size_y = bounding_box.size.y
            self.size_z = bounding_box.size.z

    def __str__(self):
        return (
            "BoundingBox " +
            "{x: " + self.x +
            ", y: " + self.y +
            ", z: " + self.z +
            ", size_x: " + self.x +
            ", size_y: " + self.y +
            ", size_z: " + self.z +
            "}"
        )


class DetectedObject:
    def __init__(self, uid, bounding_box):
        """Creates a detected object based on the bounding box and UID"""

        # Set initial values in order to call self.update()
        # because velocity is calculated based on the previous self.x and self.y.
        self.x = bounding_box.x
        self.y = bounding_box.y

        self.update(bounding_box)
        self.uid = uid

        if DEBUG:
            rospy.loginfo("Created new object: " + str(self))

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __str__(self):
        return ("DetectedObject " +
                "{UID: " + str(self.uid) +
                ", x: " + str(self.x) +
                ", y: " + str(self.y) +
                ", v_x: " + str(self.v_x) +
                ", v_y: " + str(self.v_y) +
                "}"
                )

    def update(self, bounding_box):
        """Updates the object's position, scale, calculates the velocity, resets disappearance counter.

        Parameters
        ----------
        bounding_box : BoundingBox
            A bounding box with at least x, y, z, size_x and size_y attributes.
        """
        x, y = bounding_box.x, bounding_box.y
        self.v_x = x - self.x
        self.v_y = y - self.y
        self.x = x
        self.y = y
        self.z = bounding_box.z
        self.diameter = bounding_box.size_x
        self.height = bounding_box.size_y
        self.disappeared = 0

    def has_disappeared(self):
        """Incrementing the disappearance counter every time when not visible."""
        self.disappeared += 1


class Tracker:
    """A simple tracker based on the changes of the objects' positions

    Call update() first and then get the list of current objects
    using the attribute "objects".

    Attributes
    ----------
    objects : list
        All currently tracked objects of the class DetectedObject

    Methods
    -------
    update(bounding_boxes: list)
        Updates all the objects based on the detected bounding boxes.
    get_position_dict()
        Returns a dictionary of all the existing objects' positions.
    get_velocity_dict()
        Returns a dictionary of all the existing objects' velocities.
    """

    def __init__(self, max_frames_disappeared=30, starting_id=0):
        """Initializes the tracker

        An optional disappearance threshold and starting ID for detections can be provided
        """

        self.max_frames_disappeared = max_frames_disappeared
        self.current_uid = starting_id

        self.objects = []

    def new_object(self, bounding_box):
        """Registers a newly detected object, appending it to the "objects" list."""

        self.objects.append(DetectedObject(self.current_uid, bounding_box))

        # Increment the UID for the next detected object.
        self.current_uid += 1

    def get_coordinates(self):
        """Returns coordinates of detected objects in a Python list"""

        # This can possibly be improved a bit with Numpy
        coordinates = []
        for detected_object in self.objects:
            if DEBUG:
                rospy.loginfo("Object: " + str(detected_object))
            coordinates.append((detected_object.x, detected_object.y))

        return coordinates

    def remove_if_disappeared(self, detected_object):
        """Removes the long-disappeared objects (when time disappeared >= max_frames_disappeared)

        Parameters
        ----------
        detected_object : DetectedObject
            An object whose disappearance duration is going to be checked
            (and will potentially get removed from the "objects" list).
        """
        if detected_object.disappeared >= self.max_frames_disappeared:
            self.objects.remove(detected_object)

    def update(self, bounding_boxes: list):
        """Updates the positions of detected objects, adds new objects and removes long-disappeared objects.

        Parameters
        ----------
        bounding_boxes : list
            Detected bounding boxes of type BoundingBox
        """

        # If there are no detections, all of the previous objects have disappeared.
        if len(bounding_boxes) == 0:
            for detected_object in self.objects:
                detected_object.disappeared()
                self.remove_if_disappeared(detected_object)
        # If there are no existing objects, just register all the detections.
        elif len(self.objects) == 0:
            for bounding_box in bounding_boxes:
                self.new_object(bounding_box)
        # Otherwise, find the closest matches between existing objects and detections.
        else:
            # Calculate the Euclidean distance between all the pairs of existing objects and detections.
            # Only coordinates needed for scipy.spatial.distance.cdist()
            current_coordinates = self.get_coordinates()

            # An example of the output of cdist():
            #           new_1   new_2
            # current_1   1       2
            # current_2   3       4
            distances = distance.cdist(
                np.array(current_coordinates), [(bb.x, bb.y) for bb in bounding_boxes], "euclidean")

            # The following part is mainly from the following article:
            # https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
            # Slightly modified the given code, but it was already pretty optimal,
            # using Numpy min().argsort() and argmin() for the performance-critical section.

            # Basically, this finds indices with least distances between the frames
            # current == rows; new == columns;
            indices_sorted_current = distances.min(axis=1).argsort()
            indices_sorted_new = distances.argmin(
                axis=1)[indices_sorted_current]

            # Using set() to filter out the indices already used,
            # also enables set difference operation for later
            indices_used_current = set()
            indices_used_new = set()

            for i, j in zip(indices_sorted_current, indices_sorted_new):
                if i not in indices_used_current and j not in indices_used_new:
                    # Updating the existing object's state
                    self.objects[i].update(bounding_boxes[j])
                    indices_used_current.add(i)
                    indices_used_new.add(j)

            nr_of_current_objects = distances.shape[0]
            nr_of_new_objects = distances.shape[1]

            # Dealing with the cases where the nr of
            # existing and detected objects does not match
            if nr_of_current_objects > nr_of_new_objects:
                # Finding the difference of sets == finding unused indices
                # Converting to list in order to use reversed() later
                indices_unused_current = list(set(
                    range(nr_of_current_objects)).difference(indices_used_current))

                # reversed() in order not to mess up the indexing
                # (iteration + mutation on the same list)
                for i in reversed(indices_unused_current):
                    obj = self.objects[i]
                    obj.has_disappeared()
                    self.remove_if_disappeared(obj)
            elif nr_of_current_objects < nr_of_new_objects:
                indices_unused_new = set(
                    range(nr_of_new_objects)).difference(indices_used_new)
                for j in indices_unused_new:
                    self.new_object(bounding_boxes[j])

        if VERBOSE:
            rospy.loginfo(
                "Objects:\n" + "\n".join(list(map(str, self.objects))))

    def get_position_dict(self):
        """Returns a dictionary of current objects' positions.

        Returns
        -------
        dict
            k: object's UID
            v: object's position (x and y)
        """

        position_dict = {}

        for detected_object in self.objects:
            position_dict[detected_object.uid] = (
                detected_object.x, detected_object.y)

        return position_dict

    def get_velocity_dict(self):
        """Returns a dictionary of current objects' velocities.

        Returns
        -------
        dict
            k: object's UID
            v: object's velocity (x and y speed)
        """

        velocity_dict = {}

        for detected_object in self.objects:
            velocity_dict[detected_object.uid] = (
                detected_object.v_x, detected_object.v_y)

        return velocity_dict


class RosTracker:
    def __init__(self):
        self.bounding_boxes = []
        self.max_frames_disappeared = 30
        self.time = time.time()
        self.framerate = 0

        # Initializing the tracker
        self.tracker = Tracker(self.max_frames_disappeared)

        # Initializing the predictor
        self.predictor = predictor.Predictor(self.tracker, sensitivity=0.2)

        # Defining the frame IDs
        self.frame_id_world = "world"
        self.frame_id_odom = "odom"
        self.frame_id_realsense = "realsense_mount"

        # Creating the transform broadcasters
        pose_world = Pose()
        pose_world.orientation.w = 1
        self.tf_publisher_world = rviz_util.TFPublisher(
            pose_world, "world", "odom")

        pose_realsense = Pose()
        pose_realsense.orientation = Quaternion(0.5, -0.5, 0.5, -0.5)
        pose_realsense.position.x = 0.17
        pose_realsense.position.z = 0.20
        self.tf_publisher = rviz_util.TFPublisher(
            pose_realsense, "base_link", "realsense_mount")

        # Do the initial publishing of coordinate frames
        self.tf_publisher.publish()
        self.tf_publisher_world.publish()

        # Creating the TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # For handling RViz visualization
        self.rviz = rviz_util.RViz(frame_id=self.frame_id_world)

        # Optionally, creating a list of timers (for performance measurement)
        if TIME:
            self.timers = []

        # Subscribed topics
        self.bb_point_topic = "/ros_3d_bb/bb_3d"

        # Subscribers
        self.bb_point_sub = rospy.Subscriber(
            self.bb_point_topic, BoundingBox3DArray, self.bb_callback, queue_size=1
        )

        # Published topics
        self.bounding_boxes_topic = "/ros_3d_bb/"
        self.marker_topic = "visualization_marker"

        # Publishers
        # ---

        if VERBOSE:
            rospy.loginfo("Subscribed to topic: " + self.bb_point_topic)

    def update_framerate(self):
        """Updates the program's framerate for making accurate predictions"""

        current_time = time.time()
        self.framerate = 1 / (current_time - self.time)
        self.time = current_time

    def bb_callback(self, bb_array):
        """Called when a new message containing bounding boxes is received.

        Parameters
        ----------
        bb_array : BoundingBox3DArray
            An array of bounding boxes of type BoundingBox3D
        """

        if TIME:
            timer = Timer("update")
            self.timers.append(timer)

        # A list to hold the new detections
        self.bounding_boxes = []

        # Publishing the coordinate frames
        # TODO: convert to static
        self.tf_publisher.publish()
        self.tf_publisher_world.publish()

        try:
            # Finding the transformation from the world to the RealSense camera,
            # as the detected objects should be situated in the world frame.
            transform = self.tf_buffer.lookup_transform(
                self.frame_id_world, self.frame_id_realsense, rospy.Time()
            )

            if DEBUG:
                rospy.loginfo(transform)

            for bounding_box in bb_array.boxes:
                # "do_transform_pose" requires stamped poses, so converting the original to stamped
                original_pose_stamped = PoseStamped(
                    bounding_box.header, bounding_box.center)
                transformed_pose_stamped = tf2_geometry_msgs.do_transform_pose(
                    original_pose_stamped, transform)
                # Don't transform the size vector, as the pose is already transformed!

                new_bb = BoundingBox3D(
                    bounding_box.header, transformed_pose_stamped.pose, bounding_box.size)

                if DEBUG:
                    rospy.loginfo(new_bb)

                self.bounding_boxes.append(
                    BoundingBox(bounding_box=new_bb)
                )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)

        if TIME:
            timer.update()

            self.update_framerate()
            timer.update()

            self.tracker.update(self.bounding_boxes)
            timer.update()

            self.predictor.update()
            timer.update()

            self.predictor.predict(self.framerate * 2)
            timer.update()
        else:
            self.update_framerate()
            self.tracker.update(self.bounding_boxes)
            self.predictor.update()
            self.predictor.predict(self.framerate * 2)

        # Visualization
        if VISUALIZATION:
            for obj in self.tracker.objects:
                x = obj.x
                y = obj.y
                height = obj.height
                diameter = obj.diameter
                v_x = x + obj.v_x * self.framerate
                v_y = y + obj.v_y * self.framerate
                duration = self.max_frames_disappeared / self.framerate

                # Send data from the tracker
                self.rviz.text(obj.uid, x, y, duration=duration)
                self.rviz.cylinder(obj.uid, x, y, height, diameter,
                                   duration=duration, alpha=0.5)
                self.rviz.arrow(obj.uid, x, y, v_x, v_y, duration=duration)
                # Send data from the predictor
                # Only the x and y
                predicted_x, predicted_y,  = self.predictor.predictions[obj.uid][:2]
                # obj.uid + 1000 is not probably not an ideal way to create multiple markers.
                self.rviz.arrow(obj.uid + 1000, x, y, predicted_x,
                                predicted_y, duration=duration, r=0, g=1, b=0)

            # Send the data to self.marker_topic (usually "visualization_marker")
            self.rviz.publish()

        if TIME:
            timer.stop()

    def shutdown(self):
        """If timing the code, output the final results."""

        if TIME:
            averages = Timer.average_times(self.timers)
            rospy.loginfo("Average timings (in ms): " + str(averages))
        if VERBOSE:
            rospy.loginfo("Exiting...")


def main():
    rospy.init_node("ros_3d_bb_tracker")
    ros_tracker = RosTracker()
    rospy.on_shutdown(ros_tracker.shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
