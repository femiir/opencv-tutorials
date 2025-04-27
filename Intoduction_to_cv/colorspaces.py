import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv.imread('../media/passport.jpg')

cv.imshow('Original Image', image)

# gray scale 
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

# HSV hue saturation value
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow('HSV Image', hsv)

# LAB L - lightness, A - green to red, B - blue to yellow L*a*b
lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
cv.imshow('LAB Image', lab)

# RGB
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.imshow('RGB Image', rgb)

plt.imshow(image)
plt.show()


cv.waitKey(0)