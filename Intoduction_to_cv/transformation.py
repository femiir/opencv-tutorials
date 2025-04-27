import cv2 as cv
import numpy as np

img = cv.imread('../media/passport.jpg')

cv.imshow('Passport', img)

# translation
def translate(img, x, y):
    '''
        -x --> left
        +x --> right
        -y --> up
        +y -->
    '''
    translation_matrix = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, translation_matrix, dimensions)

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)


# rotation
def rotate(img, angle, rotation_point=None):
    (height, width) = img.shape[:2]
    if rotation_point is None:
        rotation_point = (width//2, height//2)
    rotation_matrix = cv.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotation_matrix, dimensions)

rotated = rotate(img, 45)
gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)

cv.imshow('Rotated', rotated)

# flipping
flip = cv.flip(img, 1)
cv.imshow('Flip', flip)

cv.waitKey(0)