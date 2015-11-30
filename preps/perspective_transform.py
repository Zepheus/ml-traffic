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
		dir = tan(radians(self.degrees))
        (h, w, _) = im.shape

        matrixTrans = np.array([[1, 0, 0],
                                [-dir/2, 1, dir * h / 2],
                                [-dir/w, 0, 1 + dir]])

        trans = ProjectiveTransform(matrixTrans)
        imgTrans = warp(im, trans)
        if self.side != 'left':
            imgTrans = np.fliplr(imgTrans)
        return imgTrans

