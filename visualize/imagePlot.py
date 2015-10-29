from visualize import AbstractVisual
import matplotlib.pyplot as plt


class ImagePlot(AbstractVisual):

    def _prep(self,labels,data):
        im = data

        self.fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

        if labels == 'gray':
            ax.imshow(im, cmap=plt.cm.gray)
        else:
            ax.imshow(im, cmap=plt.cm.jet)
