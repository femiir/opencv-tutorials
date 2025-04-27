import cv2 as cv
import numpy as np

img = cv.imread('../media/passport.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# simple thresh holding 
threshold, thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded Image', thresh)

# inverse thresholding
threshold, thresh_inv = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse Thresholded Image', thresh_inv)

# adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 9)
cv.imshow('Adaptive Thresholded Image', adaptive_thresh)

adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow('Adaptive Thresholded Image', adaptive_thresh)


cv.waitKey(0)