import os
import matplotlib.pyplot as plt

from skimage import feature, io
from skimage.color import rgb2gray


path = os.path.join('data/train/blue_circles/D1a', '*.png')
#collection = io.imread_collection(path)
#im = collection[0]
im = io.imread('data/train/blue_circles/D1a/01178_05534.png')
img_gray = rgb2gray(im)

edges = feature.canny(img_gray, sigma=2) # only works on grayscale

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2))

ax1.imshow(im, cmap=plt.cm.jet)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)


fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()