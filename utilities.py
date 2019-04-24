import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Tuple, List

def show_images(imgs, titles=[], n_cols=3):
    n_imgs = len(imgs)
    n_titles = len(titles)
    n_rows = np.ceil(n_imgs / n_cols)

    if n_imgs != n_titles:
        titles = [""] * n_imgs

    fig = plt.figure()
    
    for i, (img, title) in enumerate(zip(imgs, titles)):
        fig.add_subplot(n_rows, n_cols, i+1)
        plt.imshow(img, cmap='binary')
        plt.title(title)
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def show_image(img):
    cv2.imshow("Frame", img)

def visualize_corners(img: np.ndarray, points: np.ndarray, radius=5, color=(255,0,0), thickness=2):
    """
    Visualizes the rgb image and the Harris corners that are detected in it.
    """
    img_copy = img.copy()
    
    for point in points.T:
        indices = point.astype(int)
        cv2.circle(img_copy, tuple(indices), radius, color, thickness)
    
    cv2.imshow("Frame", img_copy)
