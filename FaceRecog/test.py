from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import os
import numpy as np
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
        speak = Dispatch(("SAPI.SpVoice"))
        speak.Speak(str1)
        
video=cv2.VideoCapture(0)  #0 for webcam
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# to detect the faces we need to also convert frame into grayscale bcoz this cascade classifier work well on periodical img 

with open('data/names.pkl','rb') as f:
        LABELS=pickle.load(f)
with open('data/face_data.pkl','rb') as f:
        FACES=pickle.load(f)
        

# for feeding labels and face
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)
bg_img = cv2.imread("background.png")

COL_NAMES = ['NAME' , 'TIME']

while True:
    ret,frame=video.read()  #read function gives two values first boolean for camera is oky or not second is frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= facedetect.detectMultiScale(gray,1.3,5)#1.3 ,5 are thresho;d val
    for(x,y,w,h) in faces:
        crop_img= frame[y:y+h, x:x+w, :]
        resize_img = cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)   #because one image at a time
        output = knn.predict(resize_img)
        ts = time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timeStamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("attendance/attendance_"+ date + ".csv")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(50,50,255),1) #frame,axis,height width,color,thickness
        attendance = [str(output[0]),str(timeStamp)]
    bg_img[162:162+480,55:55+640]=frame
    cv2.imshow("fram",bg_img)  #window named as fram and material is frame
    k=cv2.waitKey(1) 
    #to break the loop using keyword binding function
    if k==ord('o'):
        speak("Attendance Taken!!")
        time.sleep(5)
        if exist:
                with open("attendance/attendance_"+ date + ".csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        # writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                csvfile.close()
        else:
                with open("attendance/attendance_"+ date + ".csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                csvfile.close()
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
