from visualize import AbstractVisual
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class ScatterPlot(AbstractVisual):

    def __init__(self,xlabel='x',ylabel='y',zlabel=''):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

    def _prep(self,labels,data):
        self.fig = plt.figure(1, figsize=(9, 6))

        if (self.zlabel):
            ax = self.fig.add_subplot(111,projection='3d')
        else:
            ax = self.fig.add_subplot(111)

        for label,d in zip(labels,data):
            np_data= np.array(d)
            xs = np_data[:,0]
            ys = np_data[:,1]
            if (self.zlabel):
                zs = np_data[:,2]
                ax.plot(xs,ys,zs, 'o', label=label)
            else:
                ax.plot(xs,ys, 'o', label=label)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if (self.zlabel):
            ax.set_zlabel(self.zlabel)

        plt.legend(loc='upper left')
