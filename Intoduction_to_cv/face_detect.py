import cv2 as cv
import numpy as np

#Load classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
image = cv.imread('../media/passport.jpg')
images = [cv.imread('../media/passport.jpg'), cv.imread('../media/la_familia.jpeg')]


def detect_face(images):
    """
    ** Note: This function is not very effective for detecting faces in images
    Haar cascade classifier to detect faces in images
    It is a pre-trained model that comes with OpenCV
    Easy to use and works well with frontal faces
    But not effective with side faces, faces with glasses, etc
    """
    for i, image in enumerate(images):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7) # increasing minNeighbors reduces false positives
        print(f'Number of faces found: {len(faces)}')

        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # add blur to detected faces
            blur = cv.GaussianBlur(image, (99, 99), 7)
            image[y:y+h, x:x+w] = blur[y:y+h, x:x+w]

        cv.imshow(f'Detected Faces = {len(faces)}', image)  
detect_face(images)



cv.waitKey(0)
cv.destroyAllWindows()
