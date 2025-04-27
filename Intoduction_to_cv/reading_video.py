import cv2 as cv

# read a video
capture  = cv.VideoCapture('../media/tapping.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)



    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()