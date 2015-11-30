from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


# This class represents an abstract way of visualizing data
# It provides a method for viewing and saving the resulting image.
class AbstractVisual(metaclass=ABCMeta):
    @abstractmethod
    def _prep(self, labels, data):
        pass

    def show(self, labels, data):
        self._prep(labels, data)
        plt.tight_layout()
        plt.show()

    def save(self, labels, data, saveName):
        plt.ioff()
        self._prep(labels, data)
        plt.savefig(saveName + ".png", format='png')
        plt.close(self.fig)
