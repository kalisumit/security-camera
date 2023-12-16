#This version sends only one time after the face detection
import cv2
import time
import datetime
import requests
import numpy as np
import os


message = "Someone has entered the room at "
bot_token = '6870228508:AAGNKMM6KRmUOcIsPVtqbvhRaxYkvJtg8TM'
chat_id = '-4096005044&text'

img_capture = False

cap = cv2.VideoCapture(0)                                                                           #To capure the video. We can also take video path on the place of "0"
detection = False
timer_started = False
frame_size = (int(cap.get(3))), (int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
detection_stopped_time = 60
stop_record_after_started  = 5
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #This is the file which is preTrained with millions of phtots to identify the faces in the pictures.
# fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")      #This is the file which is preTrained with millions of phtots to identify the body in the pictures.
fullbody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')


def face_recog(image_path):
    haar_cascade = cv2.CascadeClassifier('haar_face.xml')

    people = []
    for i in os.listdir(r'C:\Users\bhaga\OneDrive\Desktop\project 2\content\people'):
        people.append(i)        
    # print(people)

    features = np.load('features.npy', allow_pickle=True)
    labels = np.load('labels.npy')

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Person', gray)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')
        cv2.putText(img, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), thickness=1)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        persona_name = str(people[label])
        # cv2.imshow('Detected Face', img)            
        
        caption = f"{persona_name}\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        url = f'https://api.telegram.org/bot6870228508:AAGNKMM6KRmUOcIsPVtqbvhRaxYkvJtg8TM/sendPhoto'
        base_url = 'https://api.telegram.org/bot6870228508:AAGNKMM6KRmUOcIsPVtqbvhRaxYkvJtg8TM/sendMessage?chat_id=-4096005044&text="{}"'.format(persona_name)
        files = {'photo': open(image_path, 'rb')}
        data = {'chat_id': chat_id, 'caption': caption}
        try:
            response = requests.post(base_url, files=files, data=data)
            print("Person name sent to Telegram")
            return response
        except requests.RequestException as e:
            print(f"Error sending message and image: {e}")
            return None
    

def send_telegram_message(image_path):
    caption = f"{message}\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    url = f'https://api.telegram.org/bot6870228508:AAGNKMM6KRmUOcIsPVtqbvhRaxYkvJtg8TM/sendPhoto'
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': chat_id, 'caption': caption}
    try:
        response = requests.post(url, files=files, data=data)
        print("Message and image sent to Telegram")
        return response
    except requests.RequestException as e:
        print(f"Error sending message and image: {e}")
        return None
    
    
while True:                                                                                         #loop from where the video will start recording

    _, frame = cap.read()                                                                           #"_" is for boolean value it is use because we don't need the value of the boolean it is only for checking if the frames in the video is capturing or not. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                                  #Because this algorith only detect face in GRAY color picture we have to make the video in gray for the program.
    face = face_cascade.detectMultiScale(gray,1.3,5)
    body = fullbody_cascade.detectMultiScale(gray,1.3,5)
    upperbody = upperbody_cascade.detectMultiScale(gray,1.2,5)


    # if len(face) + len(body) + len(upperbody) > 0:
    if len(face) + len(body) > 0:
        current_time = time.time()
# It's only execute if the already video is capturing
        if detection == True: 
            timer_started = False
            image_path = f"{current_time}.png"
#Capturing image 
            if img_capture == False:
                cv2.imwrite(image_path, frame)
                print("Image Captured!!!....")
                img_capture = True
                send_telegram_message(image_path)
                face_recog(image_path)
            else:
                pass       
# It's execute when the program will run first time 
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 10, frame_size)
            print("Capturing Video!!!....")
    
    elif detection == True:
        if timer_started == True:
           if time.time() - detection_stopped_time >= stop_record_after_started :
               detection = False
               timer_started = False
               out.release()
               print("Stop Recording!!!....")
               img_capture = False   # this will help to send image after stop recording
        else:
            timer_started = True
            detection_stopped_time = time.time()
            
    if detection == True:
        out.write(frame)
            
    for(x,y,width,height) in  face:
        cv2.rectangle(frame,(x,y),(x+width , y+ height),(0,0,255), 3)
    
    for(x,y,width,height) in  body:
        cv2.rectangle(frame,(x,y),(x+width , y+ height),(0,255,0),5)
        
    for(x,y,width,height) in upperbody:
        cv2.rectangle(frame,(x,y), (x+width , y+ height),(255,0,0))
    
    cv2.imshow('Camera',frame)                                                                      # this will show the camera input on the screen
    
    if cv2.waitKey(1) == ord('q'):                                                                  #to terminate the program and for exiting the loop.
        break


out.release()
cap.release()                                                                                       #To make the camera free for other software uses.
cv2.destroyAllWindows()                                                                             #to destroy alll the window generated while runing the program.