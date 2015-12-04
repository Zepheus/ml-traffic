from features import AbstractFeature
from preps import PrepCombiner, BWTransform, ResizeTransform, IntegralTransform
import numpy as np
from concurrent.futures import *
import os
import re


# Class responsible for calculation of Haar features of image
class HaarFeature(AbstractFeature):
    # Initialise the Haar feature class with a size value indicating the size of the images
    # after resizing (default 40), a number of values indicating the sizes of the Haars (default
    # [2, 4, 5, 8, 10, 20]), a defined number of Haars to be used as features (default 20) and
    # a value indicating whether or not to use Haar features stored in a file
    def __init__(self, size=40, haar_sizes=(2, 4, 5, 8, 10, 20), n_haars=20, use_cached=True):
        self.haar_sizes = haar_sizes
        self.transform = PrepCombiner([ResizeTransform(size=size), BWTransform(), IntegralTransform()])
        self.size = size
        self.haars = [self._haar1, self._haar2, self._haar3]
        self.n_haars = n_haars
        self.use_cached = use_cached

        # Import the haar features calculated in the past for future reuse
        if use_cached:
            self.haar_configs = []
            if not os.path.exists('haarImportance.txt'):
                raise Exception("No cached file available. Please create a haarImportance.txt file first")

            with open('haarImportance.txt') as file:
                i = 0
                pattern = re.compile(
                    '\[size=(?P<size>[0-9]*)\]\[x=(?P<x>[0-9]*)\]\[y=(?P<y>[0-9]*)\]\[type=(?P<type>[0-9]*)\]'
                )
                for line in file:
                    t = re.match(pattern, line)
                    size = int(t.group("size"))
                    x = int(t.group("x"))
                    y = int(t.group("y"))
                    haar_type = int(t.group("type"))

                    self.haar_configs.append((size, x, y, haar_type))

                    if i == self.n_haars:
                        break
                    i += 1

    # Process and return the Haar features for the image
    def process(self, im):
        scaled = im.prep(self.transform)
        w, h = scaled.shape

        features = []
        if self.use_cached:
            for size, x, y, haar_type in self.haar_configs:
                sub = scaled[x:x + size, y:y + size]
                features.append(self.haars[haar_type](sub))

            return features
        else:
            # Threaded execution for performance
            executor = ThreadPoolExecutor(max_workers=5)
            tasks = []
            for s in self.haar_sizes:
                tasks.append(executor.submit(self._process_with_size, scaled, w, h, s, self.haars))
            wait(tasks)
            features = [t.result() for t in tasks]

            return np.hstack(features)

    # Method that calculates different Haars for every region of the image
    @staticmethod
    def _process_with_size(im, w, h, size, haars):
        features = []
        for x in range(w - size):
            for y in range(h - size):
                sub = im[x:x + size, y:y + size]
                features.append([haar(sub) for haar in haars])
        return np.ravel(features)

    # Calculation of Haar feature 1:
    # Calculate sum of left half and right half of the image and subtract.
    def _haar1(self, sub):
        w, h = sub.shape
        center = int(w / 2)
        part1 = self._area(sub, (0, 0), (center, h))
        part2 = self._area(sub, (center, 0), (w, h))
        return part1 - part2

    # Calculation of Haar feature 2:
    # Calculate sum of upper half and lower half of the image and subtract.
    def _haar2(self, sub):
        w, h = sub.shape
        center = int(h / 2)
        part1 = self._area(sub, (0, 0), (w, center))
        part2 = self._area(sub, (0, center), (w, h))
        return part1 - part2

    # Calculation of Haar feature 3:
    # Calculate sum of left upper corner with the right lower corner of the image and
    # the sum of the right upper corner with the left lower corner of the image and subtract
    def _haar3(self, sub):
        w, h = sub.shape
        center_w = int(w / 2)
        center_h = int(h / 2)
        part1 = self._area(sub, (0, 0), (center_w, center_h)) + self._area(sub, (center_w, center_h), (w, h))
        part2 = self._area(sub, (0, center_h), (center_w, h)) + self._area(sub, (center_w, 0), (w, center_h))
        return part1 - part2

    # Calculate sum of area in image
    @staticmethod
    def _area(sub, upper_left=None, bottom_right=None):
        if upper_left is None:
            upper_left = (0, 0)
        if bottom_right is None:
            bottom_right = sub.shape

        return sub[bottom_right[0] - 1, bottom_right[1] - 1] + sub[upper_left[0], upper_left[1]] \
               - sub[upper_left[0], bottom_right[1] - 1] - sub[bottom_right[0] - 1, upper_left[1]]
