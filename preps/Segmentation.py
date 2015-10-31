from preps import AbstractPrep
from preps import BWTransform
from skimage.filters import sobel
from skimage.feature import canny
from skimage.segmentation import slic
from skimage.draw import ellipse
from skimage.morphology import watershed
from skimage.morphology import *
from skimage import exposure
import numpy as np
from scipy import ndimage as ndi
from visualize import ImagePlot

class Segmentation(AbstractPrep):

    def process(self,im):
        #segmented = np.zeros_like(im)
        #w,h,_ = im.shape
        segm = slic(im,n_segments=5)
        #values = np.unique(segm)

        #for v in values:
        #    mask = segm == v
        #    maskDim = np.array([(x,x,x) for x in np.nditer(mask)]).reshape(im.shape)
        #    masked = np.multiply(im,maskDim)
        #    reds = np.sum(masked[:,:,0]) / np.sum(mask)
        #    greens = np.sum(masked[:,:,1]) / np.sum(mask)
        #    blues = np.sum(masked[:,:,2]) / np.sum(mask)
        #    average = (reds,greens,blues)
        #    for x in range(w):
        #        for y in range(h):
        #            if mask[x,y] == 1:
        #                segmented[x,y] = average

        closed = closing(segm)

        return closed
