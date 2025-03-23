import pathlib
import cv2 as cv
import numpy as np

people = ['femiir', 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

BASE_DIR = pathlib.Path(__file__).parent
image_dir = BASE_DIR.parent / 'media' / 'people'

features = [] # image arrays of the faces
labels = [] # the name/label of the person

def create_train():
    for person in people:
        path = image_dir / person
        label = people.index(person)

        # print('The person: ', person)
        # print('Reading images from: ', path)
        # print('Label: ', label)

        for image_path in path.iterdir():
            # skip hidden files and only process images because i use a macbook
            # you can extend list of image extensions to include more
            if image_path.name.startswith('.') or image_path.suffix not in ['.jpg', '.jpeg', '.png']:
                continue

            img = cv.imread(str(image_path))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # detect faces
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w] # face region of interest
                features.append(face_roi)
                labels.append(label)
create_train()
print('Training done')
# print('Number of features: ', len(features))
# print('Number of labels: ', len(labels))

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

# Create the models directory if it doesn't exist
model_dir = BASE_DIR.parent / 'models'
model_dir.mkdir(exist_ok=True, parents=True)  # Creates directory if it doesn't exist

face_recognizer.save(str(model_dir / 'face_trained.yml'))
np.save(str(model_dir / 'features.npy'), features)
np.save(str(model_dir / 'labels.npy'), labels)
print('Model trained successfully')


