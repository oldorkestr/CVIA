# Laboratory 2: Filters and Features

# 1 Gaussian blur using pillow
print("Gaussian blur using pillow")
from PIL import Image
from PIL import ImageFilter
img = Image.open("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Nasa.png")
blur_img = img.filter(ImageFilter.GaussianBlur(5))
blur_img.show()
input("Press Enter to continue...")

# 2 Gaussian blur using skimage
print("Gaussian blur using skimage")
from skimage import io
from skimage import filters
img = io.imread("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Nasa.png")
out = filters.gaussian(img, sigma=5)
io.imshow(out)
io.show()
input("Press Enter to continue...")

# 3 Median filter using pillow
print("Median filter using pillow")
from PIL import Image
from PIL import ImageFilter
img = Image.open("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Nasa.png")
blur_img = img.filter(ImageFilter.MedianFilter(7))
blur_img.show()
input("Press Enter to continue...")

# 4 Median filter using skimage
print("Median filter using skimage")
from skimage import io
from skimage import filters
img = io.imread("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Nasa.png")
out = filters.median(img, cval = 7)
io.imshow(out)
io.show()
input("Press Enter to continue...")

# 5 Erosion
print("Erosion")
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

# read the image file
image = plt.imread("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Linux.png")

# convert to grayscale
image = np.mean(image, axis=-1)

# binarize the image
threshold = 0.5
image = (image > threshold).astype(np.int64)

# perform binary erosion
eroded_image = morphology.binary_erosion(image)

# plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(eroded_image, cmap='gray')
ax[1].set_title('Eroded Image')
plt.show()
input("Press Enter to continue...")


# 6 Dilation
print("Dilation")
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

# read the image file
image = plt.imread("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Linux.png")

# convert to grayscale
image = np.mean(image, axis=-1)

# binarize the image
threshold = 0.5
image = (image > threshold).astype(np.int32)

# perform binary dilation
dilated_image = morphology.binary_dilation(image)

# plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(dilated_image, cmap='gray')
ax[1].set_title('Dilated Image')
plt.show()
input("Press Enter to continue...")

# 7 Custom filters

print("Custom filters")
from PIL import ImageFilter
kernel = ImageFilter.Kernel((3,3), [1,2,3,4,5,6,7,8,9])
from PIL import Image
from PIL import ImageFilter
img = Image.open("/Users/orestonyshchenko/Desktop/University/CVIA/Lab2/Nasa.png")
img = img.convert("L")
new_img = img.filter(ImageFilter.Kernel((3,3),[1,0,-1,5,0,-5,1,0,1]))
new_img.show()
input("Press Enter to continue...")