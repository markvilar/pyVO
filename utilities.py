import numpy as np
import cv2
import csv

from typing import Tuple, List

def visualize_harris_corners(img: np.ndarray, points_and_response: List[Tuple[float, np.ndarray]], radius=4, thickness=2):
    """
    Visualizes the rgb image and the Harris corners that are detected in it.
    :param img: The image.
    :param points_and_response: The response and image coordinates of the Harris corners.
    :param radius: The radius of the circles drawn around the Harris corners.
    :param thickness: The thickness of the circles drawn around the Harris corners.
    """
    img_copy = img.copy()
    
    for response, point in points_and_response:
        cv2.circle(img_copy, tuple(point), radius, (255,255,255), thickness)
    
    cv2.imshow("Harris Corners", img_copy)

def visualize_lucas_kanade(target_img: np.ndarray, warped_img: np.ndarray, error_img: np.ndarray, size=(215, 215)):
    """
    Visualizes the target image, warped image and error image in the Lucas-Kanade method.
    :param target_img: The target image.
    :param warped_img: The warped image.
    :param error_img: The error image.
    :param size: The size of the images after resizing.
    """
    target_img = cv2.resize(src=target_img, dsize=size)
    warped_img = cv2.resize(src=warped_img, dsize=size)
    error_img = cv2.resize(src=error_img*error_img, dsize=size)

    img_stack = np.concatenate((target_img, warped_img, error_img), axis=1)

    cv2.imshow("LK", img_stack)
    cv2.waitKey(250)

def read_convergence_log(file_name):
    """
    Loads a .csv file containing the number of steps used for successful
    convergence of the LK method.
    :param file_name: The file name.
    """
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        cumulative_convergence_steps = 0
        n_readings = 0
        for row in csv_reader:
            n_readings += 1
            cumulative_convergence_steps += int(row['n_convergence_steps'])

        print("Convergence log for {} ...".format(file_name))
        print("Average amount of convergence steps: {}".format(cumulative_convergence_steps/n_readings))
