import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('../media/passport.jpg')

b,g,r = cv.split(img)

cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
# ax1.imshow(b, cmap='gray')
# ax1.set_title('Blue Channel')
# ax2.imshow(g, cmap='gray')
# ax2.set_title('Green Channel')
# ax3.imshow(r, cmap='gray')
# ax3.set_title('Red Channel')

# plt.show()

# merge the channels
merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

# merge the channels with a different color
merged = cv.merge([r, g, b])
cv.imshow('Merged with different color', merged)

blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

cv.waitKey(0)