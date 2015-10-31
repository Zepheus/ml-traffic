import math

import numpy as np
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny

from skimage.transform import resize

from features import AbstractFeature
from preps import BWTransform, EqualizerTransform, PrepCombiner


class DetectCircle(AbstractFeature):
    def __init__(self, sigma=3, max_resized=64):
        self.sigma = sigma
        self.max_resized = max_resized
        self.transform = PrepCombiner([BWTransform(),EqualizerTransform()])

    def process(self, im):
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

        if len(accums) == 0: #TODO: fix, should not happen
            return [0]

        idx = np.argmax(accums)
        return [accums[idx]]
