# Laboratory 1: Introduction to Image Processing

# Image processing libraries

# Pillow

# Getting started with pillow
print("Getting started with pillow")

# 1 Reading an image

print("Reading an image")
from PIL import Image 
img = Image.open("Nasa.png") # Example reading an Image
img.show()
input("Press Enter to continue...")


# 2 Writing or saving an image

print("Writing or saving an image")
img.save("temp.png") # Example showing relative path

# 3 Cropping an image

print("Cropping an image")
dim = (100,100,400,400) #Dimensions of the ROI
crop_img = img.crop(dim)
crop_img.show()
crop_img.save("NasaCroped.png")
input("Press Enter to continue...")

# 4 Changing between color spaces

print("Changing between color spaces")
grayscale = img.convert("L")
grayscale.show()
grayscale.save("NasaGrayscalse.png")
input("Press Enter to continue...")

# 5 Geometrical transformation

# Resize
print("Resize")
resize_img = img.resize((200,200))
resize_img.show()
input("Press Enter to continue...")

# Rotate
print("Rotate")
rotate_img = img.rotate(90)
rotate_img.show()
input("Press Enter to continue...")

# 6 Image enhancement

from PIL import ImageEnhance 

# Change brightness of an image
print("Change brightness of an image")
enhancer = ImageEnhance.Brightness(img)
enhancer.enhance(2).show()
input("Press Enter to continue...")

# Change the contrast of the image
print("Change the contrast of the image")
enhancer = ImageEnhance.Contrast(img)
enhancer.enhance(2).show()
input("Press Enter to continue...")

# 7 Accessing pixels of an image

# getpixel(): This function returns the color value of the pixel at the (x, y) 
# coordinate. It takes a tuple as an argument and returns a tuple of color values
print("Getting pixels using getpixel() method")
print(img.getpixel((100,100)))
input("Press Enter to continue...")

# putpixel(): This function changes the color value of the pixel at the (x, y) 
# coordinate to a new color value. Both the coordinates and the new color value 
# are passed as an argument to the function. If the image has more than one band 
# of colors, then a tuple is passed as an argument to the function:
print("Getting pixels using putpixel() method")
img.putpixel((100,100),(20,230,145))
print(img.getpixel((100,100)))
input("Press Enter to continue...")

# Getting started with scikit-image
print("Getting started with scikit-image")


# 1 Reading an image

print("Reading an image")
from skimage import io
img = io.imread("Nasa.png")
io.imshow("Nasa.png")
io.show()
input("Press Enter to continue...")

# 2 Writing/saving an image

print("Writing/saving an image")
img = io.imread("Nasa.png")
io.imsave("temp1.png", img)
io.show()
input("Press Enter to continue...")

# 3 Data module

print("Data module")
from skimage import data
#a
io.imshow(data.camera())
io.show()
input("Press Enter to continue...")

#b
io.imshow(data.text())
io.show()
input("Press Enter to continue...")

# 4 Color module

print("Color module")

#Convert RGB to gray:
print("Convert RGB to gray")
from skimage import io, color
img = data.astronaut()
gray = color.rgb2gray(img)
io.imshow(gray)
io.show()
input("Press Enter to continue...")

#Convert RGB to HSV
print("Convert RGB to HSV")
from skimage import data
img = data.astronaut()
img_hsv = color.rgb2hsv(img)
io.imshow(img_hsv)
io.show()
input("Press Enter to continue...")

# 5 Draw module

print("Draw module")

#Circles
import numpy as np
from skimage import io, draw
print("Drawing circle")
img = np.zeros((100, 100), dtype=np.uint8)
x , y = draw.circle_perimeter(50, 50, 10)
img[x, y] = 1
io.imshow(img)
io.show()
input("Press Enter to continue...")

#Ellipses
print("Drowing ellipse")
img = np.zeros((100, 100), dtype=np.uint8)
x , y = draw.ellipse(50, 50, 10, 20)
img[x, y] = 1
io.imshow(img)
io.show()
input("Press Enter to continue...")

#Polygons
print("Drowing polygons")
img = np.zeros((100, 100), dtype=np.uint8)
r = np.array([10, 25, 80, 50])
c = np.array([10, 60, 40, 10])
x, y = draw.polygon(r, c)
img[x, y] = 1
io.imshow(img)
io.show()
input("Press Enter to continue...")
