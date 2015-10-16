from visualize import AbstractVisual
import matplotlib.pyplot as plt

class BoxPlot(AbstractVisual):

    def show(self,labels,data,saveName=""):
        # Create a figure instance
        fig = plt.figure(1, figsize=(9, 6))

        ax = fig.add_subplot(111)

        ## add patch_artist=True option to ax.boxplot()
        ## to get fill color
        bp = ax.boxplot(data, patch_artist=True)

        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set( color='#7570b3', linewidth=2)
            # change fill color
            box.set( facecolor = '#1b9e77' )

        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)

        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        ax.set_xticklabels(labels,rotation='vertical')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        #ax.set_aspect(1/len(labels))

        if saveName:
            plt.savefig(saveName + ".png", format='png')

        plt.tight_layout()
        plt.show()

