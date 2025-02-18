import cv2

face = cv2.imread('/boyz.jpg')
image_grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontal_face_default.xml')

faces = face_cascade.detectMultiScale(image_grey, scaleFactor = 1.05, minNeighbors = 1)

# for (x, y, w, h) in faces:
#   face_region = face[y:y+h, x:x+w]

#   cv2.imshow('Face Region', face_region)

#   eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
