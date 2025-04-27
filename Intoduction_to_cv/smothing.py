import cv2 as cv
import numpy as np

img = cv.imread('../media/passport.jpg')
cv.imshow('Original Image', img)

# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average Blur', average)

average = cv.blur(img, (7,7))
cv.imshow('Average Blur 7x7', average)

average = cv.blur(img, (3,7))
cv.imshow('Average Blur 3x7', average)

# Gaussian Blur
gaussian = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian Blur', gaussian)

gaussian = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian Blur 7x7', gaussian)

# Median Blur 
# Median blur is used to remove noise from the image
# It is used to remove salt and pepper noise
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# bilateral blur
# Bilateral blur is used to remove noise while keeping the edges sharp and in focus
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)