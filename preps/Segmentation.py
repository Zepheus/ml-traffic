from preps import AbstractPrep
from preps import BWTransform
from skimage.filters import sobel
from skimage.feature import canny
from skimage.segmentation import slic, quickshift
from skimage.draw import ellipse
from skimage.morphology import watershed
from skimage.morphology import *
from skimage import exposure
import numpy as np
from scipy import ndimage as ndi
from visualize import ImagePlot

class Segmentation(AbstractPrep):

    def process(self,im):
        return slic(im,n_segments=5)
