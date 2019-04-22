from pyflann import *
import numpy as np
import cv2

dataset = np.array(
        [
            [1., 1, 1, 2, 3],
            [10, 10, 10, 3, 2],
            [100, 100, 2, 30, 1]
            ])

testset = np.array(
        [
            [1., 1, 1, 1, 1],
            [90, 90, 10, 10, 1]
            ])

flann = FLANN()
result, dists = flann.nn(
        dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)

img = np.zeros((512, 512, 3), np.uint8)
#cv2.line(img, (0,0), (511,511), (255,0,0), 5)
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destoyAllWindows()

print(result)
print(dists)
