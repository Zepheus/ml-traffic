from features import AbstractFeature
from preps import HsvTransform, PrepCombiner, BWTransform
from skimage.measure import regionprops, label
import numpy as np

from visualize import ImagePlot


class HsvFeature(AbstractFeature):
    def __init__(self):
        self.transform = PrepCombiner([HsvTransform()])

    def process(self, im):
        ImagePlot().show("", im.image)

        gray_img = BWTransform().process(im.image)
        # Filtered white image
        filtered_img = gray_img > 0.5
        ImagePlot().show("", filtered_img)
        # Divide in regions
        labeled_img = label(filtered_img)
        ImagePlot().show("", labeled_img)
        # Filter out largest white region
        mult_img = np.multiply(labeled_img, filtered_img) # only whitish regions are considered
        props = regionprops(mult_img)
        largest_region_img = np.zeros_like(mult_img)
        max_region = props[np.argmax([region.area for region in props])]
        for coord in max_region.coords:
            largest_region_img[coord[0], coord[1]] = 1
        ImagePlot().show("", largest_region_img)
        # Exclude white regions on the border
        excluded_border_image = np.ones_like(mult_img)
        width, height = excluded_border_image.shape
        for region in props:
            remove = False
            for coord in region.coords:
                if coord[0] == 0 or coord[1] == 0 or coord[0] == width - 1 or coord[1] == height - 1:
                    remove = True
                    break
            if remove:
                for coord in region.coords:
                    excluded_border_image[coord[0], coord[1]] = 0
        excluded_border_image *= filtered_img
        ImagePlot().show("", excluded_border_image)
        # Remove small regions
        exclude_small_img = np.zeros_like(mult_img)
        filtered_props = [region for region in props if region.area > 0.05 * width * height]
        for region in filtered_props:
            for coord in region.coords:
                exclude_small_img[coord[0], coord[1]] = 1
        ImagePlot().show("", exclude_small_img)
        # Only most central white region

        # Combine images
        result = (exclude_small_img + excluded_border_image + largest_region_img) >= 3
        ImagePlot().show("TADA!", result)

        # hAvg = np.mean(hsvIm[:, :, 0])
        # hWhiteValue = hsvIm[:, :, 2] > 0.6
        # hWhiteSaturation = hsvIm[:, :, 1] < 0.25
        # whiteMask = np.logical_and(hWhiteValue, hWhiteSaturation)
        # hWhiteAvg = np.mean(whiteMask)
        #
        # hB = im.image[:, :, 2] < 60
        # hG = im.image[:, :, 1] < 60
        # hR = im.image[:, :, 0] < 60
        # blackMask = np.logical_and(hR, np.logical_and(hB, hG))
        # hBlackAvg = np.mean(blackMask)
