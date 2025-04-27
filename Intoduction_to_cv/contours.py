import cv2 as cv
import numpy as np

# Load image
image = cv.imread('../media/passport.jpg')

if image is None:
    print('Could not read the image')
    exit()

cv.imshow('Original Image', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

# Canny edge detection
canny = cv.Canny(gray, 125, 175)
color_canny = cv.Canny(image, 125, 175)
cv.imshow('Canny Edges', canny)
cv.imshow('Color Canny Edges', color_canny)

# Find contours
contours_canny, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours_canny)} contours found!')

#blur the image
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Blur Grayed', blur)

blurred_canny = cv.Canny(blur, 125, 175)
contours_blurred_canny, hierarchies = cv.findContours(blurred_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours_blurred_canny)} contours found!')

# Thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)
contours_thresh, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours_thresh)} contours found!')

# create a blank image
blank = np.zeros(image.shape, dtype='uint8')
cv.imshow('Blank Image', blank)

# Draw contours
cv.drawContours(blank, contours_thresh, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)

cv.drawContours(blank, contours_canny, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn on Image', blank)




cv.waitKey(0)

