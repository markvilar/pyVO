import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Tuple, List

def show_images(imgs, frame_titles=[]):
    for img, title in zip(imgs, frame_titles):
        cv2.imshow(title, img)

def show_image(img):
    cv2.imshow("Frame", img)

def visualize_corners(img: np.ndarray, points_and_response: List[Tuple[float, np.ndarray]], radius=4, color=(255,0,0), thickness=2):
    """
    Visualizes the rgb image and the Harris corners that are detected in it.
    """
    img_copy = img.copy()
    
    for response, point in points_and_response:
        cv2.circle(img_copy, tuple(point), radius, color, thickness)
    
    cv2.imshow("Harris Corners", img_copy)
