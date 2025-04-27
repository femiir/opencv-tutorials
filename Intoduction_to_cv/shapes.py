import cv2 as cv
import numpy as np


# draw a blank image 
blank = np.zeros((500,500, 3), dtype='uint8')
# cv.imshow('Blank', blank) # show the blank image

# paint on the image
blank[:] = 0,255,0 
# cv.imshow('Green', blank) # show the green image

# draw in just a section of the image
blank[200:300, 300:400] = 0,0,255
cv.imshow('Red', blank) # show the red image

# draw a rectangle
# cv.rectangle(blank, (0,0), (250,250), (0,0, 250), thickness=2)
#alternatively
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,0, 250), thickness=cv.FILLED)
cv.imshow('Rectangle', blank)


# draw a circle 
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (255,0,0), thickness=3)
cv.imshow('Circle', blank)

# draw a line
cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
cv.imshow('Line', blank)

# add text
cv.putText(blank, 'femiir', (20,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0), thickness=2)
cv.imshow('Text', blank)


cv.waitKey(0)