from features import AbstractFeature
from preps import RatioTransform
from skimage.measure import regionprops, label
from skimage import exposure
import numpy as np

from visualize import ImagePlot


class HsvFeature(AbstractFeature):
    def process(self, img):
        # Extract blue regions
        rgb2yuv = np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]])
        yuv = self.rescale_values(np.dot(img.image, rgb2yuv.T))
        ImagePlot().show("gray", yuv[:, :, 2])
        white_img = self.extract_color(yuv[:, :, 0])
        red_img = self.extract_color(yuv[:, :, 2])
        blue_img = self.extract_color(yuv[:, :, 1])
        # Test if overlap
        red_white_overlap = red_img * white_img
        ratio_img = RatioTransform().process(img.image)
        if np.sum(red_white_overlap) > 0:
            RGB_avg = self.relative_proportions(red_white_overlap, ratio_img)
            if np.abs(RGB_avg[0] - RGB_avg[1]) <= 0.2 and np.abs(RGB_avg[0] - RGB_avg[2]) <= 0.2:
                red_img -= red_white_overlap
            else:
                white_img -= red_white_overlap
        black_img = self.extract_color(1 - yuv[:, :, 0])
        ImagePlot().show("gray", white_img)
        ImagePlot().show("gray", red_img)
        ImagePlot().show("gray", blue_img)

    def rescale_values(self, img):
        img[:, :, 1] = (img[:, :, 1] + 0.436)/(0.436 * 2)
        img[:, :, 2] = (img[:, :, 2] + 0.615)/(0.615 * 2)
        return img

    def relative_proportions(self, bw_img, img):
        sum = np.sum(bw_img)
        RGB_avg = [0, 0, 0]
        for i in range(len(RGB_avg)):
            mult_img = bw_img * img[:, :, i]
            RGB_avg[i] = np.sum(mult_img) / sum
        return RGB_avg / RGB_avg[0] # Normalize

    def extract_color(self, img, show=False):
        ratio_img = RatioTransform().process(img)
        width, height = ratio_img.shape
        # if exposure.is_low_contrast(ratio_img, fraction_threshold=0.45):
        ratio_img = exposure.equalize_hist(ratio_img)
        # gray_img *= (im.image[:, :, 0] > 150) * (im.image[:, :, 1] > 150) * (im.image[:, :, 2] > 150)
        # Find an appropriate threshold in order to extract white-like regions
        min_threshold = np.amin(ratio_img)
        max_threshold = np.amax(ratio_img)
        value_range = (max_threshold - min_threshold)
        threshold = value_range * 0.5 + min_threshold
        props = []
        max_iter = 1
        while (len(props) <= 2 or len(props) >= 10) and max_iter < 1000:
            # Filtered white image
            filtered_img = ratio_img > threshold
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
            threshold = threshold + value_range * 0.01 if too_less_regions else threshold - value_range * 0.01
            max_iter += 1
        # Extract largest white-like region
        print("Threshold: %.2f" % threshold)
        largest_region_img = np.zeros_like(ratio_img)
        if len(props) == 0:
            return largest_region_img
        max_region_coords = props[np.argmax([region.area for region in props])].coords
        largest_region_img[max_region_coords[:, 0], max_region_coords[:, 1]] = 1
        # Exclude white regions on the border
        excluded_border_img = np.ones_like(ratio_img)
        border_region_coords = np.array([coord.tolist() for coords in
                                         [region.coords for region in props if 0 in region.coords or
                                          width - 1 in region.coords[:, 0] or height - 1 in region.coords[:, 1]]
                                         for coord in coords])
        if len(border_region_coords) != 0:
            excluded_border_img[border_region_coords[:, 0], border_region_coords[:, 1]] = 0
        excluded_border_img *= filtered_img
        # Remove small regions
        exclude_small_img = np.zeros_like(ratio_img)
        excluded_small_coords = np.array([coord.tolist() for region in [region for region in props if region.area > 0.01*width*height]
                          for coord in region.coords])
        exclude_small_img[excluded_small_coords[:, 0], excluded_small_coords[:, 1]] = 1
        # Only most central white region
        cx = width/2
        cy = height/2
        dx = width/5
        dy = height/5
        center_region_img = np.zeros_like(ratio_img)
        for region in props:
            keep = False
            for coord in region.coords:
                if cx - dx <= coord[0] <= cx + dx and cy - dy <= coord[1] <= cy + dy:
                    keep = True
                    break
            if keep:
                for coord in region.coords:
                    center_region_img[coord[0], coord[1]] = 1
        # # Exclude regions with high saturation
        # hsv_img = HsvTransform().process(ratio_img)
        # filtered_hsv_img = hsv_img[:, :, 1] < 0.5
        # Combine images
        result = (exclude_small_img + excluded_border_img + largest_region_img + center_region_img) >= 3
        if show:
            print("Show of largest region")
            ImagePlot().show("", largest_region_img)
            print("Show of exclusion border")
            ImagePlot().show("", excluded_border_img)
            print("Show of removal small regions")
            ImagePlot().show("", exclude_small_img)
            print("Show of retaining center regions")
            ImagePlot().show("", center_region_img)
            # print("Filtered hsv image")
            # ImagePlot().show("", filtered_hsv_img)
            print("Show of result")
            ImagePlot().show("TADA!", result)

        return result