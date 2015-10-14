import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import data, color, io
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


# Load picture and detect edges
im = io.imread('data/train/blue_circles/D1b/00638_01854.png')
img_gray = rgb2gray(im)
edges = canny(img_gray, sigma=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

# Detect two radii
# calculate image diameter
shape = im.shape
diam = math.sqrt(shape[0]**2 + shape[1]**2)
hough_res = hough_circle(edges, np.arange(diam/3, diam * 0.8, 2))


centers = []
accums = []
radii = []

for radius, h in zip(hough_radii, hough_res):
    peaks = peak_local_max(h, num_peaks=1)
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius] * num_peaks)

# Draw the most prominent 5 circles
for idx in np.argsort(accums)[::-1][:5]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(center_y, center_x, radius)
    #image[cy, cx] = (220, 20, 20)

ax.imshow(im, cmap=plt.cm.jet)