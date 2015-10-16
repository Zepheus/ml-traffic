from features import AbstractFeature
from preps import HsvTransform
import numpy as np

class HsvFeature(AbstractFeature):

    def __init__(self):
        self.transform = HsvTransform()


    def process(self,im):
        hsvIm = self.transform.process(im)
        hAvg = np.mean(hsvIm[:,:,0])
        sAvg = np.mean(hsvIm[:,:,1])
        vAvg = np.mean(hsvIm[:,:,2])
        test = np.mean(hsvIm,axis=2)
        return [hAvg,sAvg,vAvg]
