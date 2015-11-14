from preps import AbstractPrep
from skimage import transform
import math
from visualize import ImagePlot


class RotateTransform(AbstractPrep):
    def __init__(self, degrees, cropImage=True):
        self.degrees = degrees
        self.crop = cropImage

# Reference: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    def rotatedRectWithMaxArea(self, w, h, angle):
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return wr, hr

    def process(self, im):
        if self.crop:
            (h, w, _) = im.shape
            nw, nh = self.rotatedRectWithMaxArea(w, h, math.radians(self.degrees))
            rotated = transform.rotate(im, self.degrees, resize=True)
            (rh, rw, _) = rotated.shape

            image_size = (rw, rh)
            image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

            x1 = int(image_center[0] - nw * 0.5)
            x2 = int(image_center[0] + nw * 0.5)
            y1 = int(image_center[1] - nh * 0.5)
            y2 = int(image_center[1] + nh * 0.5)

            rotated_cropped = rotated[y1:y2, x1:x2, :]
            return rotated_cropped
        else:
            return transform.rotate(im, self.degrees, resize=True)
