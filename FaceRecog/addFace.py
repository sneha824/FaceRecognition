import cv2
import pickle,os
import numpy as np
video=cv2.VideoCapture(0)  #0 for webcam
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# to detect the faces we need to also convert frame into grayscale bcoz this cascade classifier work well on periodical img 

face_data=[]
i=0
name=input("Enter your name: ")
while True:
    ret,frame=video.read()  #read function gives two values first boolean for camera is oky or not second is frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= facedetect.detectMultiScale(gray,1.3,5)#1.3 ,5 are thresho;d val
    for(x,y,w,h) in faces:
        crop_img= frame[y:y+h, x:x+w, :]
        resize_img = cv2.resize(crop_img,(50,50))
        if len(face_data)<=50 and i%10==0:  #to take image after 10 frame..i%10 is used
            face_data.append(resize_img)
        i=i+1
        # how many picture we have taken we can see that
        cv2.putText(frame,str(len(face_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        # passing frame , len putText can take str so converted , origin, font, font scale,fontcolor,thickness
        cv2.rectangle(frame,(x,y), (x+w,y+h),(50,50,255),1) #frame,axis,height width,color,thickness
    cv2.imshow("fram",frame)  #window named as fram and material is frame
    k=cv2.waitKey(1)    #to break the loop using keyword binding function
    if k==ord('q') or len(face_data)==50:
        break
video.release()
cv2.destroyAllWindows()

# converting data into numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape(50,-1)


# collecting data

# for creating names.pkl file in data folder
if 'names.pkl' not in os.listdir('data/'):
    names=[name]*50
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
        
else:
    with open('data/names.pkl','rb') as f:
        names=pickle.load(f)
    names=names+[name]*50
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
        
# for creating face data.pkl
if 'face_data.pkl' not in os.listdir('data/'):
    # names=[name]*50
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(face_data,f)
        
else:
    with open('data/face_data.pkl','rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces,face_data,axis=0)
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(names,f)
        
