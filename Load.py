import os
from skimage import io
from sklearn import cross_validation

class LabelledImage():

    def __init__(self,image,label,superLabel=''):
        self.image = image
        self.label = label
        self.superLabel = superLabel


def load(directories):
    values = []
    for directory in directories:
        for dirpath, dirnames, _ in os.walk(directory):
            if not dirnames:
                images = io.imread_collection(os.path.join(dirpath, '*.png'))
                label = os.path.basename(dirpath)
                superLabel = os.path.basename(os.path.dirname(dirpath))
                for (image, fn) in zip(images, images.files):
                    values.append(LabelledImage(image,label,superLabel))

    # permute array

    return values

def folds(images,fitter,k=1):
    x = [li.image for li in images]
    y = [li.superlabel for li in images]
    return cross_validation.cross_val_score(fitter, x, y, cv=k, n_jobs=2)

