import numpy as np

from features import AbstractFeature
from preps import HsvTransform
from preps import RatioTransform
from visualize import ImagePlot
from skimage import color as c
from skimage import exposure

class ColorCenter(AbstractFeature):

    def __init__(self):
        self.transform = HsvTransform()
        self.ratioTransform = RatioTransform()

    def process(self,im):

        p2, p98 = np.percentile(im, (2, 98))
        img_rescale = exposure.rescale_intensity(im, in_range=(p2, p98))
        hsvImage = self.transform.process(img_rescale)
        blackPositions = []
        whitePositions = []
        bluePositions = []
        redPositions = []

        for x in range(hsvImage.shape[0]):
            for y in range(hsvImage.shape[1]):
                color = hsvImage[x,y]

                if (color[1] <= 0.3):
                    # black
                    if (color[2] < 0.5):
                        blackPositions.append([x,y])
                        #hsvImage[x,y] = [0,0,0]
                    # white
                    if (color[2] >= 0.5):
                        #hsvImage[x,y] = [0,0,1]
                        whitePositions.append([x,y])
                else:
                    pass
                    if (color[0] > 0.5 and color[0] < 0.9 and color[2] >= 0.4):
                        #hsvImage[x,y] = [0.7,1,1]
                        bluePositions.append([x,y])
                    if (color[0] > 0.9 or color[0] < 0.3 and color[2] >= 0.4):
                        #hsvImage[x,y] = [0,1,1]
                        redPositions.append([x,y])


        #ImagePlot().show(c.hsv2rgb(hsvImage))

        if (len(blackPositions) == 0):
            blackPositions.append([0,0])
        if (len(whitePositions) == 0):
            whitePositions.append([0,0])
        if (len(bluePositions) == 0):
            bluePositions.append([0,0])
        if (len(redPositions) == 0):
            redPositions.append([0,0])

        blackCenter = np.divide(np.mean(blackPositions,axis=0),hsvImage.shape[0:2])
        whiteCenter = np.divide(np.mean(whitePositions,axis=0),hsvImage.shape[0:2])
        blueCenter = np.divide(np.mean(bluePositions,axis=0),hsvImage.shape[0:2])
        redCenter = np.divide(np.mean(redPositions,axis=0),hsvImage.shape[0:2])
        return np.concatenate([blackCenter,whiteCenter,blueCenter,redCenter])