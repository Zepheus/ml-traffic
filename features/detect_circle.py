import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import data, color, io
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


from features import AbstractFeature

class DetectCircle(AbstractFeature):

    def __init__(self):
        pass

    def process(self,im):
        img_gray = rgb2gray(im)

        success = False

        sig = 0.5
        while not success:
            edges = canny(img_gray, sigma=sig)

            # Detect two radii
            # calculate image diameter
            shape = im.shape
            diam = math.sqrt(shape[0]**2 + shape[1]**2)
            radii = np.arange(diam/3, diam * 0.8, 2)
            hough_res = hough_circle(edges, radii)

            # peaks = peak_local_max(hough_res, num_peaks=1, min_distance=1, threshold_abs=0.01, exclude_border=True)
            accums = []

            for radius, h in zip(radii, hough_res):
                # For each radius, extract two circles
                peaks = peak_local_max(h, num_peaks=1, min_distance=1)
                success = len(peaks) > 0
                if success:
                    accums.extend(h[peaks[:, 0], peaks[:, 1]])
                else:
                    print('Adjusting sigma...')
                    sig /= 2 # backoff sigma
                    break

            if success:
                idx = np.argsort(accums)[::-1][1]
                return accums[idx]



