import numpy as np
import cv2

from typing import Tuple

fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7


def project_points(ids: np.ndarray, points: np.ndarray, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the 2D points to 3D using the depth image and the camera instrinsic parameters.
    :param ids: A N vector point ids.
    :param points: A 2xN matrix of 2D points
    :param depth_img: The depth image. Divide pixel value by 5000 to get depth in meters.
    :return: A tuple containing a N vector and a 3xN vector of all the points that where successfully projected.
    """
    # points = [u, v]
    depth_img = depth_img / 5000

    # Flip points to get [v, u]
    img_indices = np.flip(points, axis=0)

    point_depths = depth_img[tuple(img_indices.astype(int))]

    mask = point_depths > 0
    valid_ids = ids[mask]
    valid_points = points[:,mask]

    Z = point_depths[mask]
    X = (valid_points[0] - cx) * (Z / fx)
    Y = (valid_points[1] - cy) * (Z / fy)

    valid_points_3d = np.stack([X, Y, Z], axis=0)

    printing = False
    
    if printing:
        print("img_indices: {} \n{}\n".format(img_indices.shape, img_indices))
        print("ids: {} \n{}\n".format(ids.shape, ids))
        print("points: {} \n{}\n".format(points.shape, points))
        print("point_depths: {} \n{}\n".format(point_depths.shape, point_depths))
        print("valid_ids: {} \n{}\n".format(valid_ids.shape, valid_ids))
        print("valid_points: {} \n{}\n".format(valid_points.shape, valid_points))
        print("X: {} \n{}\n".format(X.shape, X))
        print("Y: {} \n{}\n".format(Y.shape, Y))
        print("Z: {} \n{}\n".format(Z.shape, Z))

    return valid_ids, valid_points_3d
