from preps import AbstractPrep


class CropTransform(AbstractPrep):

    def __init__(self, size=8, position='left'):
        self.size = size
        self.position = position

    def process(self, im):
        (h, w, _) = im.shape
        if self.position == 'left':
            return im[:, self.size:, :]
        elif self.position == 'right':
            return im[:, :w-self.size, :]
        elif self.position == 'top':
            return im[self.size:, :, :]
        elif self.position == 'bottom':
            return im[:h-self.size, :, :]
        else:
            return im
