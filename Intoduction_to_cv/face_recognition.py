import cv2 as cv
import numpy as np
import pathlib

# Get the directory where the script is located
BASE_DIR = pathlib.Path(__file__).parent


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

people = ['femiir', 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']


face_recognizer.read(str(BASE_DIR.parent / 'models' / 'face_trained.yml'))
features = np.load(str(BASE_DIR.parent / 'models' / 'features.npy'), allow_pickle=True)
labels = np.load(str(BASE_DIR.parent / 'models' / 'labels.npy'))


img = cv.imread(str(BASE_DIR.parent / 'media' / 'untrained' / '6.png'))
# img = cv.imread(str(BASE_DIR.parent / 'media' / 'la_familia.jpeg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

detected_faces = []
for (x, y, w, h) in faces_rect:
    face_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(face_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    detected_faces.append({'rect': (x, y, w, h), 'label': label, 'confidence': confidence})
   

if detected_faces:
    best_face = min(detected_faces, key=lambda x: x['confidence'])
    
    for face in detected_faces:
        x, y, w, h = face['rect']
        label = face['label']
        confidence = face['confidence']
    
        if face == best_face:
            text = f'{people[label]} {confidence}'
            color = (0, 255, 0)
        else:
            text = f'Unknown {confidence}'
            color = (0, 0, 255)

        cv.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
        cv.putText(img, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv.imshow('Detected Faces', img)
cv.waitKey(0)

      



cv.imshow(f'Detected Face as {people[label]} with a confidence of {confidence}', img)
cv.waitKey(0)


    
