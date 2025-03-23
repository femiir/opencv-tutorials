import cv2 as cv


img = cv.imread('../media/passport.jpg')

# convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blur the image
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# edge cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)
# blurred_canny = cv.Canny(blur, 125, 175) # reduce edges in the blurred image
# cv.imshow('Blurred Canny Edges', blurred_canny)

# dilating the image
dilated = cv.dilate(canny, (3,3), iterations=1)
cv.imshow('Dilated', dilated)
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated high', dilated)

# eroding
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('Eroded', eroded)

# resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA) # shrink the image
resized = cv.resize(img, (500,500), interpolation=cv.INTER_LINEAR) # when increasing the size of the image
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC) # when increasing the size of the image
cv.imshow('Resized', resized)

# cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)