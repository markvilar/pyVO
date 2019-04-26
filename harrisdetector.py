import cv2
import numpy as np

from operator import itemgetter
from typing import Tuple, List

from utilities import show_images, show_image, visualize_corners

def harris_corners(img: np.ndarray, threshold=100.0, blur_sigma=6.0, blur_kernel_size = 3, k=0.06, nms_bin_size=13, visualize=True) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :param k: Harris response trace scale factor.
    :param nms_bin_size: Size of the bins used to perform non-maximum suppression.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    # img[v,u], y-axis parallel to v-avis, x-axis parallel to u-axis
    # img_height, img_width = img.shape

    if nms_bin_size < 3:
        nms_bin_size = 3
    elif nms_bin_size % 2 == 0:
        nms_bin_size += 1

    if blur_kernel_size < 3:
        blur_kernel_size = 3

    nms_kernel = np.ones((nms_bin_size, nms_bin_size))
    blur_kernel = (blur_kernel_size, blur_kernel_size)
    
    # TODO: Look into multi-threading
    # Calculate x- and y-derivative of the image using Scharr
    img_x = cv2.Scharr(src=img, ddepth=-1, dx=1, dy=0)
    img_y = cv2.Scharr(src=img, ddepth=-1, dx=0, dy=1)

    # Blur products of derivatives
    img_xx = cv2.GaussianBlur(img_x * img_x, blur_kernel, blur_sigma)
    img_yy = cv2.GaussianBlur(img_y * img_y, blur_kernel, blur_sigma)
    img_xy = cv2.GaussianBlur(img_x * img_y, blur_kernel, blur_sigma)

    # Compute Harris response
    dets = img_xx * img_yy - img_xy * img_xy
    traces = img_xx + img_yy
    responses = dets - k * traces * traces

    # Perform non-maximum suppression
    is_not_maxima = np.logical_not(responses == cv2.dilate(responses, nms_kernel, iterations=1))
    responses[is_not_maxima] = threshold

    # Threshold based on response value
    mask = responses > threshold
    filtered_positions = np.flip(np.argwhere(mask), axis=1)
    filtered_responses = responses[mask]

    # Sort responses and corners based on response value
    sorting_indices = np.flip(np.argsort(filtered_responses))
    sorted_positions = np.take(filtered_positions, sorting_indices, axis=0)
    sorted_responses = np.take(filtered_responses, sorting_indices)

    corners = [(sorted_responses[i], sorted_positions[i]) for i in range(len(sorted_positions))]

    if visualize:
        visualize_corners(img, corners)

    return corners
