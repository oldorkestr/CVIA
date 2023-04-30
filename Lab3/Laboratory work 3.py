# Laboratory 3: Drilling Deeper into Features - Object Detection

# 1 Harris corner detection
print("Harris corner detection")
import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import data, feature
import matplotlib.pyplot as plt

image = data.astronaut()  # Load sample image
image = rgb2gray(image)

corners = feature.corner_harris(image)  # Extract Harris corners
# Apply Harris corner detection
corner_response = corner_harris(corners)

# Find peaks in the response image
corner_coords = corner_peaks(corner_response, min_distance=5)

# Plot the original image and the detected corners
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(corner_coords[:, 1], corner_coords[:, 0], 'o', color='red', markersize=5)
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
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage import data
from skimage.color import rgb2gray

# Load images
img_left = rgb2gray(data.astronaut())[50:250, 50:250]
img_right = rgb2gray(data.astronaut())[50:250, 300:500]

# Initialize ORB detector
orb = ORB(n_keypoints=200)

# Detect keypoints and compute descriptors
orb.detect_and_extract(img_left)
keypoints_left = orb.keypoints
descriptors_left = orb.descriptors

orb.detect_and_extract(img_right)
keypoints_right = orb.keypoints
descriptors_right = orb.descriptors

# Match descriptors between images
matches = match_descriptors(descriptors_left, descriptors_right)

# Plot results
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

plt.gray()

plot_matches(ax, img_left, img_right, keypoints_left, keypoints_right, matches)

plt.show()
input("Press Enter to continue...")

# # 4 oFAST – FAST keypoint orientation
print("oFAST – FAST keypoint orientation")
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color, data
from skimage.feature import corner_fast, corner_orientations, plot_matches

# Load image
img = data.astronaut()
img_gray = color.rgb2gray(img)

# Detect corners using FAST algorithm
keypoints = corner_fast(img_gray, n=12, threshold=0.15)

# Calculate orientations for each keypoint
orientations = corner_orientations(img_gray, keypoints)

# Plot keypoints with orientations
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(keypoints[:, 1], keypoints[:, 0], '+r', markersize=5)
for i, angle in enumerate(orientations):
    x, y = keypoints[i]
    r = 5
    dx = r * np.cos(angle)
    dy = r * np.sin(angle)
    ax.plot([y - dx, y + dx], [x - dy, x + dy], '-g', linewidth=1)

ax.axis((0, img.shape[1], img.shape[0], 0))
plt.show()

input("Press Enter to continue...")

# # 5 FAST detector
# print("FAST detector")

# input("Press Enter to continue...")

# # 6 Orientation by intensity centroid
# print("Orientation by intensity centroid")

# input("Press Enter to continue...")

# # 7 rBRIEF – Rotation-aware BRIEF
# print("rBRIEF – Rotation-aware BRIEF")

# input("Press Enter to continue...")

# # 8 Steered BRIEF
# print("Steered BRIEF")

# input("Press Enter to continue...")

# # 9 Variance and correlation
# print("Variance and correlation")

# input("Press Enter to continue...")

# # 10 Image stitching
# print("Image stitching")

# input("Press Enter to continue...")
