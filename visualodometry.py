import cv2
import time

from dataloader import DataLoader
from harrisdetector import harris_corners
from pointTracker import PointTracker
from pointProjection import project_points

from debug.PointsVisualizer import PointVisualizer
from utilities import visualize_corners

dl = DataLoader('dataset/rgbd_dataset_freiburg1_rpy') # Edit this string to load a different dataset

tracker = PointTracker()
vis = PointVisualizer()

# Set initial position of cameras in visualizer
initial_orientation, initial_position = dl.get_transform()
vis.set_groundtruth_transform(initial_orientation, initial_position)
vis.set_estimated_transform(initial_orientation, initial_position)

# STEP 1
# Get points for the first frame
grey_img = dl.get_greyscale()
depth_img = dl.get_depth()
points_and_response = harris_corners(grey_img)
tracker.add_new_corners(grey_img, points_and_response)

# Project the points in the first frame
previous_ids, previous_points = tracker.get_position_with_id()
previous_ids, previous_points = project_points(previous_ids, previous_points, depth_img)
vis.set_projected_points(previous_points, initial_orientation, initial_position)

current_orientation = initial_orientation
current_position = initial_position

while dl.has_next():
    time_total_start = time.time()
    dl.next()

    # Visualization
    gt_orientation, gt_position = dl.get_transform()
    vis.set_groundtruth_transform(gt_orientation, gt_position)

    # Get images
    grey_img = dl.get_greyscale()
    depth_img = dl.get_depth()
    
    # STEP 2 and 3
    # Track current points on new image
    time_tracker_start = time.time()
    tracker.track_on_image(grey_img)
    time_tracker_stop = time.time()
    tracker.visualize(grey_img)

    # Project tracked points
    ids, points = tracker.get_position_with_id()
    ids, points = project_points(ids, points, depth_img)
    vis.set_projected_points(points, gt_orientation, gt_position)

    # STEP 4
    # Replace lost points
    time_harris_start = time.time()
    points_and_response = harris_corners(grey_img)
    time_harris_stop = time.time()
    tracker.add_new_corners(grey_img, points_and_response)

    # Visualization
    #visualize_corners(grey_img, points_and_response)

    time_total_stop = time.time()

    time_tracker = time_tracker_stop - time_tracker_start
    time_harris = time_harris_stop - time_harris_start
    time_total = time_total_stop - time_total_start
    print("Time spent: {:>8}, FPS: {:>8}".format(time_total, 1/time_total))
    print("Tracker: {:>8}".format(time_tracker))
    print("Harris: {:>8}".format(time_harris))
    cv2.waitKey(20)
    
cv2.destroyAllWindows()
