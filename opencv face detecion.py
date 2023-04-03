import cv2
from tensorflow import keras
import numpy as np

face_deection_model = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')
model = keras.models.load_model("age_gender.h5")

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_deection_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        print(x, y)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150))

        face = np.expand_dims(face, axis=0)
        face = face / 255

        output = model.predict(face)

        age = output[0][0]
        gender = np.round(output[1][0])
        gender_type = 'male' if gender == 0 else 'female'
        text = "Age - " + str(age) + " Gender - " + gender_type 

        img = cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        print(output)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff

    if(k == 27):
        break


cap.release()

cv2.destroyAllWindows()
