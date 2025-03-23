import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('../media/passport.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# grayscale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])



# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.xticks(range(0, 256, 10))
# plt.show()

# mask
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(gray, gray, mask=mask)
cv.imshow('Masked Image', masked)

gray_hist_masked = cv.calcHist([gray], [0], mask, [256], [0, 256])

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(gray_hist_masked)
axs[0].set_title('Grayscale Histogram Masked')
axs[0].set_xlabel('Bins')
axs[0].set_ylabel('# of pixels')

axs[1].plot(gray_hist)
axs[1].set_title('Grayscale Histogram')
axs[1].set_xlabel('Bins')
axs[1].set_ylabel('# of pixels')



plt.show()

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')



# color histogram
def plot_color_hist(img, mask=None):
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv.calcHist([img], [i], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

plot_color_hist(img)

# color mask
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Colored Masked Image', masked)

plot_color_hist(img, mask)
cv.waitKey(0)