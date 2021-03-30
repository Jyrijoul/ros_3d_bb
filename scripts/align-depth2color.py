## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2.pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3 #3 meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    """
    Convert the depth and image point information to metric coordinates
    Parameters:
    -----------
    depth 	 	 	 : double
                           The depth value of the image point
    pixel_x 	  	 	 : double
                           The x value of the image coordinate
    pixel_y 	  	 	 : double
                            The y value of the image coordinate
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    X : double
        The x value in meters
    Y : double
        The y value in meters
    Z : double
        The z value in meters
    """
    d_offset = -4.2
    x_offset = -15
    X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx * (depth + d_offset) + x_offset
    Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy * (depth + d_offset)
    return X, Y, depth

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        profile = pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        depth_intrinsics = depth_profile.get_intrinsics()

        def get_point(x, y, depth):
            p = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            return p
        # x, y = 142, 418
        # x, y = 373, 625
        # x, y = 153, 149
        # x, y = 79, 129  # -685, -300, 1670
        # x, y = 327, 228
        x, y = 582, 277
        # aligned_depth_frame.get_intrinsics()
        depth = depth_image[y, x]
        point = get_point(x, y, depth)
        point_ = convert_depth_pixel_to_metric_coordinate(depth, x, y, depth_intrinsics)
        # print(np.shape(color_image), np.shape(depth_image))
        print(point)
        print(point_)
        print(depth_intrinsics)

        # pixel = rs.rs2_project_point_to_pixel(depth_intrinsics, point)
        # print(pixel)

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
