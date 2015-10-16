from visualize import AbstractVisual
import matplotlib.pyplot as plt


class ImagePlot(AbstractVisual):

    def show(self,data):
        im = data

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

        ax.imshow(im, cmap=plt.cm.jet)
        plt.show()
