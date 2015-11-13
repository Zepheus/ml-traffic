from features import AbstractFeature
from preps import HsvTransform, PrepCombiner, BWTransform
from skimage.measure import regionprops, label
from skimage import exposure
import numpy as np

from visualize import ImagePlot


class HsvFeature(AbstractFeature):
    def __init__(self):
        self.transform = PrepCombiner([HsvTransform()])

    def process(self, im):
        self.extract_white_regions(im)

    def extract_white_regions(self, im):
        width, height, _ = im.image.shape
        gray_img = BWTransform().process(im.image)
        # if exposure.is_low_contrast(im.image, fraction_threshold=0.45):
        gray_img = exposure.equalize_hist(gray_img)
        # gray_img *= (im.image[:, :, 0] > 150) * (im.image[:, :, 1] > 150) * (im.image[:, :, 2] > 150)
        # Find an appropriate threshold in order to extract white-like regions
        threshold = (np.amax(gray_img) - np.amin(gray_img)) * 0.5
        min_threshold = np.amin(gray_img)
        max_threshold = np.amax(gray_img)
        props = []
        max_iter = 1
        while (len(props) <= 2 or len(props) >= 10) and max_iter < 1000:
            # Filtered white image
            filtered_img = gray_img > threshold
            # Divide in regions
            labeled_img = label(filtered_img) + 1
            # Filter out largest white region
            mult_img = np.multiply(labeled_img, filtered_img) # only whitish regions are considered
            props = regionprops(mult_img)
            props = [region for region in props if region.area > 0.01 * width * height]
            # Adjust threshold boundaries and threshold
            too_less_regions = len(props) <= 2
            min_threshold = threshold if too_less_regions else min_threshold
            max_threshold = threshold if not too_less_regions else max_threshold
            threshold = threshold + 0.01 if too_less_regions else threshold - 0.01
            max_iter += 1
        # Extract largest white-like region
        largest_region_img = np.zeros_like(gray_img)
        max_region_coords = props[np.argmax([region.area for region in props])].coords
        largest_region_img[max_region_coords[:, 0], max_region_coords[:, 1]] = 1
        print("Show of largest region")
        ImagePlot().show("", largest_region_img)
        # Exclude white regions on the border
        excluded_border_img = np.ones_like(gray_img)
        border_region_coords = np.array([coord.tolist() for coords in
                                         [region.coords for region in props if 0 in region.coords or
                                          width - 1 in region.coords[:, 0] or height - 1 in region.coords[:, 1]]
                                         for coord in coords])
        excluded_border_img[border_region_coords[:, 0], border_region_coords[:, 1]] = 0
        excluded_border_img *= filtered_img
        print("Show of exclusion border")
        ImagePlot().show("", excluded_border_img)
        # Remove small regions
        exclude_small_img = np.zeros_like(gray_img)
        excluded_small_coords = np.array([coord.tolist() for region in [region for region in props if region.area > 0.01*width*height]
                          for coord in region.coords])
        exclude_small_img[excluded_small_coords[:, 0], excluded_small_coords[:, 1]] = 1
        print("Show of removal small regions")
        ImagePlot().show("", exclude_small_img)
        # Only most central white region
        cx = width/2
        cy = height/2
        dx = width/5
        dy = height/5
        center_region_img = np.zeros_like(gray_img)
        for region in props:
            keep = False
            for coord in region.coords:
                if cx - dx <= coord[0] <= cx + dx and cy - dy <= coord[1] <= cy + dy:
                    keep = True
                    break
            if keep:
                for coord in region.coords:
                    center_region_img[coord[0], coord[1]] = 1
        print("Show of retaining center regions")
        ImagePlot().show("", center_region_img)
        # Combine images
        result = (exclude_small_img + excluded_border_img + largest_region_img + center_region_img) >= 3
        print("Show of result")
        ImagePlot().show("TADA!", result)