import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import os.path
from cv2 import WINDOW_NORMAL


ESC = 27

def start_webcam():
   # Charger le modèle à partir du fichier JSON
   json_file = open('top_models\\fer.json', 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   model = model_from_json(loaded_model_json)

   # Chargez les poids et les modéliser
   model.load_weights('top_models\\fer.h5')
   face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   cap = cv2.VideoCapture(0)

   while True:
       ret, img = cap.read()
       if not ret:
           break

       gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

       for (x, y, w, h) in faces_detected:
           cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
           roi_gray = gray_img[y:y + w, x:x + h]
           roi_gray = cv2.resize(roi_gray, (48, 48))
           img_pixels = image.img_to_array(roi_gray)
           img_pixels = np.expand_dims(img_pixels, axis=0)
           img_pixels /= 255.0

           predictions = model.predict(img_pixels)
           max_index = int(np.argmax(predictions))

           emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
           predicted_emotion = emotions[max_index]

           cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

           resized_img = cv2.resize(img, (1000, 700))
           cv2.imshow('Reconnaissance des émotions faciales', resized_img)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()

def analyze_picture(path):
    
    model_emotion=cv2.face.FisherFaceRecognizer_create()
    model_emotion.read('models/emotion_classifier_model.xml')
    
    model_gender=cv2.face.FisherFaceRecognizer_create()
    model_gender.read('models/gender_classifier_model.xml')
    
    
    
    
    
    
    
    window_size=(1280, 720)
    window_name = "Facifier Static (press ESC to exit)"
    
    
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    image = cv2.imread(path, 1)
    
    normalized_faces = []
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
    coordinates =  faceCascade.detectMultiScale( image, scaleFactor=1.1, minNeighbors=15, minSize=(70, 70)  )
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
    
    for face in cropped_faces:
     face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
     face = cv2.resize(face, (350, 350))
     normalized_faces.append(face)
    
    
    find_faces=zip(normalized_faces, coordinates)

    for normalized_face, (x, y, w, h) in find_faces:
        emotion_prediction = model_emotion.predict(normalized_face)
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == ESC:
        cv2.destroyWindow(window_name)

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

   





    # Utiliser un modèle pour prédire
    # Main programme :
        
    choice = input("Voulez-vous utiliser la camera ?(oui/non) ")
    if (choice == 'oui'):
        start_webcam()
    elif (choice == 'non'):
        run_loop = True
        
        print("Le chemin par défaut est défini sur data/sample/")
        print("Tapez q ou quitter pour terminer le programme")
        while run_loop:
            path = "../data/sample/"
            file_name = input("Spécifiez le fichier image : ")
            if file_name == "q" or file_name == "quitter":
                run_loop = False
            else:
                path += file_name
                if os.path.isfile(path):
                    analyze_picture(path)
                else:
                    print("Fichier introuvable!")
    else:
        print("Saisie invalide, sortie du programme.")

