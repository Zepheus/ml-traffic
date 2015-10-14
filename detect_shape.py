import os
import matplotlib.pyplot as plt
import math
from skimage import feature, io
from skimage.measure import label, regionprops
from skimage.color import rgb2gray


#path = os.path.join('data/train/blue_circles/D1a', '*.png')
#collection = io.imread_collection(path)
#im = collection[0]

im = io.imread('data/train/blue_circles/D1a/01178_05534.png')
img_gray = rgb2gray(im)

edges = feature.canny(img_gray, sigma=2) # only works on grayscale
labeled_image = label(edges)

#Locating center of each region
regions = regionprops(labeled_image)
if len(regions) == 0:
    raise ValueError('no regions detected')

max_area_size = max([x.area for x in regions])
max_region = next(x for x in regions if x.area == max_area_size)
print('Max region size: ', max_region.area)
print('Eccentricity: ', max_region.eccentricity)

# display results
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 1))
ax1.imshow(im, cmap=plt.cm.jet)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

# show max region
y0, x0 = max_region.centroid
orientation = max_region.orientation
x1 = x0 + math.cos(orientation) * 0.5 * max_region.major_axis_length
y1 = y0 - math.sin(orientation) * 0.5 * max_region.major_axis_length
x2 = x0 - math.sin(orientation) * 0.5 * max_region.minor_axis_length
y2 = y0 - math.cos(orientation) * 0.5 * max_region.minor_axis_length

#ax1.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
#ax1.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
#ax1.plot(x0, y0, '.g', markersize=15)

minr, minc, maxr, maxc = max_region.bbox
bx = (minc, maxc, maxc, minc, minc)
by = (minr, minr, maxr, maxr, minr)
ax1.plot(bx, by, '-b', linewidth=2.5)


#ax2.imshow(edges, cmap=plt.cm.gray)
#ax2.axis('off')
#ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()