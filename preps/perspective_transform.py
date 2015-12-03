from preps import AbstractPrep
import numpy as np
from skimage.transform import ProjectiveTransform, warp
from math import tan, radians


# this preprocessor apply a perspective transform to the image
# The area farther away will be projected smaller.
# The area closer will be projected larger.
class PerspectiveTransform(AbstractPrep):

    def __init__(self, degrees=12, side='left'):
        self.degrees = degrees
        self.side = side

    def process(self, im):
        # if side is right flip so it becomes right
        if self.side != 'left':
            im = np.fliplr(im)

        # slope of the perspective
        slope = tan(radians(self.degrees))
        (h, w, _) = im.shape

        matrix_trans = np.array([[1, 0, 0],
                                [-slope/2, 1, slope * h / 2],
                                [-slope/w, 0, 1 + slope]])

        trans = ProjectiveTransform(matrix_trans)
        img_trans = warp(im, trans)
        if self.side != 'left':
            img_trans = np.fliplr(img_trans)
        return img_trans

