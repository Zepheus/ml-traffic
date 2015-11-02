from features import AbstractFeature
from preps import PrepCombiner, BWTransform, ResizeTransform, IntegralTransform
import numpy as np
from concurrent.futures import *

class HaarFeature(AbstractFeature):

    def __init__(self, size=40,haarSizes=[2,4,5,8,10,20]):
        checks = [size % x for x in haarSizes]
        i = np.nonzero(checks)[0] # array is 1D so only index 0 is relevant
        if (len(i) != 0):
            k = ", ".join(map(str,np.array(haarSizes)[i]))
            raise Exception("the following haarSizes are not compatible with the given size: %s" % k)

        self.haarSizes = haarSizes
        self.transform = PrepCombiner([ResizeTransform(size=size),BWTransform(),IntegralTransform()])

    def process(self, im):
        scaled = im.prep(self.transform)
        w,h = scaled.shape

        haars = [self._haar1,self._haar2,self._haar3]
        features = []
        executor = ThreadPoolExecutor(max_workers=5)
        tasks = []
        for s in self.haarSizes:
            tasks.append(executor.submit(self._processWithSize,scaled,w,h,s,haars))
        wait(tasks)
        features = [t.result() for t in tasks]

        return np.hstack(features)

    def _processWithSize(self,im,w,h,size,haars):
        features = []
        for x in range(w-size):
            for y in range(h-size):
                sub = im[x:x+size,y:y+size]
                features.append([h(sub) for h in haars])
        return np.ravel(features)


    def _haar1(self,sub):
        w,h = sub.shape
        center = int(w/2)
        part1 = self._area(sub,(0,0),(center,h))
        part2 = self._area(sub,(center,0),(w,h))
        return part1-part2

    def _haar2(self,sub):
        w,h = sub.shape
        center = int(h/2)
        part1 = self._area(sub,(0,0),(w,center))
        part2 = self._area(sub,(0,center),(w,h))
        return part1-part2

    def _haar3(self,sub):
        w,h = sub.shape
        centerw = int(w/2)
        centerh = int(h/2)
        part1 = self._area(sub,(0,0),(centerw,centerh)) + self._area(sub,(centerw,centerh),(w,h))
        part2 = self._area(sub,(0,centerh),(centerw,h)) + self._area(sub,(centerw,0),(w,centerh))
        return part1-part2


    def _area(self,sub,upperLeft=None,bottomRight=None):
        if upperLeft is None:
            upperLeft = (0,0)
        if bottomRight is None:
            bottomRight = sub.shape

        return sub[bottomRight[0]-1,bottomRight[1]-1] + sub[upperLeft[0],upperLeft[1]] \
               - sub[upperLeft[0],bottomRight[1]-1] - sub[bottomRight[0]-1,upperLeft[1]]