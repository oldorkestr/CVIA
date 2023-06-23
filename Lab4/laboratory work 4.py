# Contour detection
from skimage import measure
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt
from skimage import data, color, io

#Read an image
img = data.astronaut()

#Convert the image to grayscale
img_gray = rgb2gray(img)

#Find edges in the image
img_edges = sobel(img_gray)

#Find contours in the image
contours = measure.find_contours(img_edges, 0.2)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(img_edges, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
input("Press Enter to continue...")


# The Watershed algorithm
from sys import exit
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import  disk
from skimage.segmentation import watershed
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

img = data.astronaut()
img_gray = rgb2gray(img)

image = img_as_ubyte(img_gray)

#Calculate the local gradients of the image
#and only select the points that have a
#gradient value of less than 20
markers = rank.gradient(image, disk(5)) < 20
markers = ndi.label(markers)[0]

gradient = rank.gradient(image, disk(2))

#Watershed Algorithm
labels = watershed(gradient, markers)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box'})

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title("Markers")

ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest', alpha=.7)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
input("Press Enter to continue...")


# Superpixels
from skimage import segmentation, color
from skimage.io import imread
from skimage import graph
from matplotlib import pyplot as plt
import skimage

imgl = data.logo()
img = skimage.color.rgba2rgb(imgl)

img_segments = segmentation.slic(img, compactness=20, n_segments=500)
superpixels = color.label2rgb(img_segments, img, kind='avg')

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(img)
ax[1].imshow(superpixels)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
input("Press Enter to continue...")


from skimage import data, segmentation
from skimage import graph
img = data.astronaut()
labels = segmentation.slic(img)
rag = graph.rag_mean_color(img, labels)

out1 = color.label2rgb(labels, img, kind='avg')
new_labels = graph.cut_threshold(labels, rag, 10)


out2 = color.label2rgb(new_labels, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(img)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
input("Press Enter to continue...")
