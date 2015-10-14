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
#im = io.imread('data/train/blue_circles/D1b/02583_06123.png')
im = io.imread('data/train/rectangles_up/B21/00969_01994.png')

img_gray = rgb2gray(im)
edges = canny(img_gray, sigma=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

# Detect two radii
# calculate image diameter
shape = im.shape
diam = math.sqrt(shape[0]**2 + shape[1]**2)
radii = np.arange(diam/3, diam * 0.8, 2)
hough_res = hough_circle(edges, radii)


peaks = peak_local_max(hough_res, num_peaks=1, min_distance=1, threshold_abs=0.01, exclude_border=True)
if len(peaks) > 0:
    radius = radii[peaks[0][0]]
    x_circle = peaks[0][1]
    y_circle = peaks[0][2]

    circle = plt.Circle((x_circle, y_circle), radius=radius, fc='y', alpha=0.4)
    plt.gca().add_patch(circle)

ax.imshow(im, cmap=plt.cm.jet)
plt.show()