import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import data, color, io
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.measure import label, regionprops

from features import AbstractFeature

class RegionRatio(AbstractFeature):

    def __init__(self):
        pass

    def process(self, im):
        img_gray = rgb2gray(im)
        img_adapted = exposure.equalize_hist(img_gray)
        edges = canny(img_adapted, sigma=3)

        labeled_image = label(edges)

        #Locating regions
        regions = regionprops(labeled_image)
        if len(regions) == 0:
            return [0, 0]

        max_area_size = max([x.area for x in regions])
        max_region = next(x for x in regions if x.area == max_area_size)

        if max_region.minor_axis_length == 0: # TODO: solve div by zero
            return [max_region.major_axis_length, max_region.eccentricity]
        else:
            return [max_region.major_axis_length / max_region.minor_axis_length, max_region.eccentricity]