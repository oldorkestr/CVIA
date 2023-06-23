# Laboratory 3: Drilling Deeper into Features - Object Detection

# 1 Harris corner detection
print("Harris corner detection")
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage import data, color, io

#Harris corner detection
#Read an image
image = data.astronaut()
image = rgb2gray(image)

corners = corner_harris(image)
coords = corner_peaks(corners, min_distance=5)
coords_subpix = corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis((0, 350, 350, 0))

plt.show()

input("Press Enter to continue...")

# 2 Local Binary Patternsz
print("Local Binary Patterns")
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import data, color, io

# Load image
img = color.rgb2gray(data.astronaut())

# Define LBP parameters
radius = 3
n_points = 8 * radius
method = 'uniform'

# Calculate LBP
lbp = local_binary_pattern(img, n_points, radius, method)

# Plot results
fig, (ax_img, ax_lbp) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax_img.imshow(img, cmap=plt.cm.gray)
ax_img.set_title('Original Image')

ax_lbp.imshow(lbp, cmap=plt.cm.gray)
ax_lbp.set_title('LBP Image')

for ax in (ax_img, ax_lbp):
    ax.axis('off')

plt.show()

input("Press Enter to continue...")

# # 3 Oriented FAST and Rotated BRIEF (ORB)
print("Oriented FAST and Rotated BRIEF (ORB)")
from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#Read the original image
image_org = data.astronaut()

#Convert the image gray scale
image_org = rgb2gray(image_org)

#We prepare another image by rotating it. Only to demonstrate feature mathcing
image_rot = tf.rotate(image_org, 180)

#We create another image by applying affine transform on the image
tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -200))
image_aff = tf.warp(image_org, tform)

#We initialize ORB feature descriptor
descriptor_extractor = ORB(n_keypoints=200)

#We first extract features from the original image
descriptor_extractor.detect_and_extract(image_org)
keypoints_org = descriptor_extractor.keypoints
descriptors_org = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(image_rot)
keypoints_rot = descriptor_extractor.keypoints
descriptors_rot = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(image_aff)
keypoints_aff = descriptor_extractor.keypoints
descriptors_aff = descriptor_extractor.descriptors

matches_org_rot = match_descriptors(descriptors_org, descriptors_rot, cross_check=True)
matches_org_aff = match_descriptors(descriptors_org, descriptors_aff, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], image_org, image_rot, keypoints_org, keypoints_rot, matches_org_rot)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

plot_matches(ax[1], image_org, image_aff, keypoints_org, keypoints_aff,matches_org_aff)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")


plt.show()
input("Press Enter to continue...")

# 4 FAST detector
print("FAST detector")
from skimage.feature import ORB, match_descriptors
from skimage.io import imread
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.color import rgb2gray
from skimage.io import imsave, show
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
import numpy as np
from skimage import io
img = data.astronaut()
fig, ax= plt.subplots(nrows=1, ncols=2)
image0=img[:,0:500]
image1=img[:,100:600]
ax[0].imshow(image0)
ax[1].imshow(image1)
fig.set_size_inches(14,10)
plt.show()
input("Press Enter to continue...")
