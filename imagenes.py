#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os, sys
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

from PIL import Image, ImageDraw, ImageFont
video_capture1 = cv2.VideoCapture(0)
video_capture2 = cv2.VideoCapture(1)
video_capture3 = cv2.VideoCapture(2)
image1 = cv2.imread('1.jpg',1)
image2 = cv2.imread('2.jpg',1)
fondo = cv2.imread('davivienda.jpg',1)
re_im1= cv2.resize (image1, (300, 300))
re_fondo= cv2.resize(fondo, (700, 630))
fondo_copy= cv2.resize(fondo, (700, 630))
#face_recort = gray1[y:y+h, x:x+w]
def button_actions(event,x,y,flags,param):
	if (event == cv2.EVENT_LBUTTONDOWN and x>630 and x<690 and y>10 and y<50):
		ret, foto1 =video_capture2.read()
		re_foto1= cv2.resize (foto1, (300, 300))
		re_fondo[320:620, 5:305]=re_foto1
	if (event == cv2.EVENT_LBUTTONDOWN and x>630 and x<690 and y>10 and y<50):
		ret, foto2 =video_capture3.read()
		re_foto2= cv2.resize (foto1, (300, 300))
		re_fondo[5:305, 5:305]=re_foto2
	
	elif (event == cv2.EVENT_LBUTTONDOWN and x>630 and x<690 and y>130 and y<170):
		re_fondo[5:305,5:305]=fondo_copy[5:305,5:305]
		re_fondo[320:620, 5:305]=fondo_copy[320:620, 5:305]
		


def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None
    #num_face=0
    #for (x,y,w,h) in faces:
        #extract the face area
     #   (x, y, w, h) = faces[num_face]
      #  num_face+=
    #return only the face part of the image
    #return gray[y:y+w, x:x+h], faces[num_face]
    return faces

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
img=re_fondo
cv2.namedWindow('img')
cv2.setMouseCallback('img',button_actions)

#botones y textos
cv2.rectangle(re_fondo, (630,10), (690, 50), (20, 0, 150), -2)
cv2.putText(re_fondo, "Documento", (631,30), cv2.FONT_HERSHEY_DUPLEX, 0.33, (0, 0, 0), 0)

cv2.rectangle(re_fondo, (630,70), (690, 110), (20, 0, 150), -2)
cv2.putText(re_fondo, "Foto", (640,90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 0)

cv2.rectangle(re_fondo, (630,130), (690, 170), (20, 0, 150), -2)
cv2.putText(re_fondo, "limpiar", (631,150), cv2.FONT_HERSHEY_DUPLEX, 0.53, (0, 0, 0), 0)


while True:
	ret2, foto1= video_capture2.read()
	ret, test_img1 = video_capture1.read()
	re_im2= cv2.resize (test_img1, (300, 300))
	re_fondo[5:305,5:305]=re_im1
	re_fondo[5:305, 315:615]=re_im2
	
	cv2.imshow('img',re_fondo )  
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture1.release()
cv2.destroyAllWindows()	
