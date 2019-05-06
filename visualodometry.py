import cv2

from dataloader import DataLoader
from harrisdetector import harris_corners
from pointTracker import PointTracker
from pointProjection import project_points

from debug.PointsVisualizer import PointVisualizer

# Parameters
dataset = 'freiburg1_desk'
show_harris_corners = False# Shows the Harris corners
show_lk_method = False# Shows the target, image and error
show_n_frames = 9999999 # Number of frames to show

# Data loader
dataset_path = 'dataset/rgbd_dataset_' + dataset
dl = DataLoader(dataset_path)

# Initialization
tracker = PointTracker()
vis = PointVisualizer()

# Set initial position of cameras in visualizer
initial_orientation, initial_position = dl.get_transform()
vis.set_groundtruth_transform(initial_orientation, initial_position)
vis.set_estimated_transform(initial_orientation, initial_position)

# Get points for the first frame
grey_img = dl.get_greyscale()
depth_img = dl.get_depth()
points_and_response = harris_corners(grey_img, visualize=show_harris_corners)
tracker.add_new_corners(grey_img, points_and_response)

# Project the points in the first frame
previous_ids, previous_points = tracker.get_position_with_id()
previous_ids, previous_points = project_points(previous_ids, previous_points, depth_img)
vis.set_projected_points(previous_points, initial_orientation, initial_position)

current_orientation = initial_orientation
current_position = initial_position

frame_count = 0

while dl.has_next():
    frame_count += 1
    dl.next()

    # Visualization
    gt_orientation, gt_position = dl.get_transform()
    vis.set_groundtruth_transform(gt_orientation, gt_position)

    # Get images
    grey_img = dl.get_greyscale()
    depth_img = dl.get_depth()
    
    # Track current points on new image
    tracker.track_on_image(grey_img, visualize=show_lk_method)
    tracker.visualize(grey_img)

    # Project tracked points
    ids, points = tracker.get_position_with_id()
    ids, points = project_points(ids, points, depth_img)
    vis.set_projected_points(points, gt_orientation, gt_position)

    # Replace lost points
    points_and_response = harris_corners(grey_img, visualize=show_harris_corners)
    tracker.add_new_corners(grey_img, points_and_response)
    
    cv2.waitKey(1)

    if frame_count >= show_n_frames:
        break
    
tracker.write_convergence_log(dataset)
cv2.destroyAllWindows()
