import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List
from scipy import signal, ndimage
from utilities import show_images

def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0, k=0.04, nms_n_corners=5, nms_n_bins=10, print_result=False) -> List[Tuple[float, np.ndarray]]:
    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :param k: Harris response trace scale factor.
    :param nms_n_corners: Maximum number of corners in each nms bin.
    :param nms_n_bins: Number of nms bins along each axis of the image.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    # img[v,u], y-axis parallel to v-avis, x-axis parallel to u-axis
    img_height, img_width = img.shape

    # Calculate x- and y-derivative of the image using Sobel operators
    sobel_x = np.array([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]])
    sobel_y = np.array([[-1.0, -2.0, -1.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 1.0]])
    img_x = signal.convolve2d(img, sobel_x, mode='same', boundary='symm')
    img_y = signal.convolve2d(img, sobel_y, mode='same', boundary='symm')
    
    # Compute products of derivatives
    img_xx = np.multiply(img_x, img_x)
    img_yy = np.multiply(img_y, img_y)
    img_xy = np.multiply(img_x, img_y)
    
    # Blur products of derivatives
    blur_img_xx = ndimage.filters.gaussian_filter(img_xx, blur_sigma)
    blur_img_yy = ndimage.filters.gaussian_filter(img_yy, blur_sigma)
    blur_img_xy = ndimage.filters.gaussian_filter(img_xy, blur_sigma)

    # Set up the structure tensor
    first_row = np.stack([blur_img_xx, blur_img_xy], axis=2)
    second_row = np.stack([blur_img_xy, blur_img_yy], axis=2)
    structure_tensor = np.stack([first_row, second_row], axis=2)

    # Compute structure matrices determinants and traces
    dets = np.linalg.det(structure_tensor)
    traces = np.trace(structure_tensor, axis1=2, axis2=3)

    # Compute Harris response
    responses = dets - k*np.square(traces)

    # Threshold based on response value
    positions = np.argwhere(responses > threshold)
    responses = responses[np.where(responses > threshold)]
    
    # Sort responses and corners based on response
    filter_indices = np.flip(np.argsort(responses))
    positions = np.take(positions, filter_indices, axis=0)
    responses = np.take(responses, filter_indices)
    
    # Perform non-maximum suppression
    bin_height = int(np.ceil(img_height / nms_n_bins))
    bin_width = int(np.ceil(img_width / nms_n_bins))
    bin_size = np.array([bin_height, bin_width])
    bin_positions = (positions / bin_size).astype(int)

    nms_indices = []
    nonempty_bins = np.unique(bin_positions, axis=0)
    bins = np.full((nonempty_bins.shape[0], nms_n_corners), np.nan)

    for index, bin_position in enumerate(bin_positions):
        bin_index = np.argwhere(np.all(nonempty_bins==bin_position, axis=1))[0][0]
        bin_values = bins[bin_index]
        is_not_nan = np.logical_not(np.isnan(bin_values))

        n_neighbours = np.sum(is_not_nan, dtype=int)

        if n_neighbours < nms_n_corners:
            valid_index = np.argwhere(np.isnan(bin_values))[0][0]
            bins[bin_index][valid_index] = index
            nms_indices.append(index)

    nms_indices = np.array(nms_indices)
    nms_positions = np.take(positions, nms_indices, axis=0)
    nms_responses = np.take(responses, nms_indices)

    corners = [(nms_responses[i], nms_positions[i]) for i in range(len(nms_positions))]

    if print_result:
        print("Harris corner detector detected {} corners...".format(len(corners)))

    return corners
