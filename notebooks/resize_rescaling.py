import cv2 as cv


def rescale_frame(frame, scale=0.75):
    """
    This works for live video, images and videos majorly existing videos
    """
    width  = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def change_res(width, height):
    """
    This works for live video only
    """
    capture.set(3, width)
    capture.set(4, height)

capture =  cv.VideoCapture('../media/tapping.mp4')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescale_frame(frame, scale=0.75)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break