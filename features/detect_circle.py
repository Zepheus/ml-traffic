import math

import numpy as np
from skimage.transform import hough_circle, hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.feature import peak_local_max, canny
import matplotlib.pyplot as plt

from skimage.transform import resize

from features import AbstractFeature
from preps import BWTransform, EqualizerTransform, PrepCombiner
from visualize import ImagePlot


class DetectCircle(AbstractFeature):
    def __init__(self, sigma=2, max_resized=64):
        self.sigma = sigma
        self.max_resized = max_resized
        self.transform = PrepCombiner([BWTransform(), EqualizerTransform()])

    def _process2(self, im):
        # img = np.zeros((25, 25), dtype=np.uint8)
        # rr, cc = ellipse_perimeter(10, 10, 6, 8)
        # img[rr, cc] = 1
        # img[12:, 12:] = 0
        # result = hough_ellipse(img, threshold=8)

        # ImagePlot().show("grey", img)

        # fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True,
        #                                 subplot_kw={'adjustable': 'box-forced'})
        # result.sort(order='accumulator')

        # best = list(result[-1])
        # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        # print("ellipse found: yc %d  xc %d  a %d  b %d    accumulator: %f" % (yc, xc, a, b, best[0]))
        # orientation = best[5]
        # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        # img[cy, cx] = 2

        # ax1.set_title('Original picture')
        # ax1.imshow(img)

        # print("plotting...")
        # plt.show()
        (width, height, _) = im.image.shape

        img_adapted = im.prep(self.transform)

        if width > self.max_resized or height > self.max_resized:
            scaleHeight = self.max_resized / height
            scaleWidth = self.max_resized / width
            scale = min(scaleHeight, scaleWidth)
            img_adapted = resize(img_adapted, (int(width * scale), int(height * scale)))

        edges = canny(img_adapted, sigma=self.sigma)
        ImagePlot().show("gray", edges)

        result = hough_ellipse(edges, accuracy=10, threshold=10, max_size=width)
        result.sort(order='accumulator')

        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        print("ellipse found: yc %d  xc %d  a %d  b %d    accumulator: %f" % (yc, xc, a, b, best[0]))
        orientation = best[5]
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        img_adapted[cy, cx] = 0

        fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True,
                                        subplot_kw={'adjustable': 'box-forced'})

        ax1.set_title('Original picture')
        ax1.imshow(img_adapted)

        edges[cy, cx] = 2

        ax2.set_title('Edge (white) and result (red)')
        ax2.imshow(edges)

        print("plotting...")
        plt.show()

        return [best[0]]

    def process(self, im):
        return self._process2(im)

        (width, height, _) = im.image.shape

        img_adapted = im.prep(self.transform)

        if width > self.max_resized or height > self.max_resized:
            scaleHeight = self.max_resized / height
            scaleWidth = self.max_resized / width
            scale = min(scaleHeight, scaleWidth)
            img_adapted = resize(img_adapted, (int(width * scale), int(height * scale)))

        edges = canny(img_adapted, sigma=self.sigma)

        # Detect two radii
        # calculate image diameter
        shape = im.image.shape
        diam = math.sqrt(shape[0] ** 2 + shape[1] ** 2)
        radii = np.arange(diam / 3, diam * 0.8, 2)
        hough_res = hough_circle(edges, radii)

        accums = []
        for radius, h in zip(radii, hough_res):
            # For each radius, extract two circles
            peaks = peak_local_max(h, num_peaks=1, min_distance=1)
            if len(peaks) > 0:
                accums.extend(h[peaks[:, 0], peaks[:, 1]])

        if len(accums) == 0:  # TODO: fix, should not happen
            return [0]

        idx = np.argmax(accums)
        return [accums[idx]]
