import cv2 as cv
import numpy as np

img = cv.imread('../media/room.jpeg')
image = cv.imread('../media/passport.jpg')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = cv.Laplacian(_gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)


# sobelx = np.uint8(np.absolute(sobelx))
# sobely = np.uint8(np.absolute(sobely))
# sobel = np.uint8(np.absolute(sobel))

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)