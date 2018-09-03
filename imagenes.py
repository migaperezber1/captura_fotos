#import OpenCV module

import cv2
#import os module for reading training data directories and paths
import os, sys
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

from PIL import Image, ImageDraw, ImageFont
video_capture1 = cv2.VideoCapture(0) #selfie
video_capture2 = cv2.VideoCapture(1) #documento
fondo = cv2.imread('davivienda.jpg',1)
bot=cv2.imread('boton.jpg',1)
log=cv2.imread('logo.jpg',1)
logo=cv2.resize(log,(130,50))
boton=cv2.resize(bot,(100,60))
re_fondo= cv2.resize(fondo, (700, 630))
fondo_copy= cv2.resize(fondo, (700, 630))
#face_recort = gray1[y:y+h, x:x+w]
def button_actions(event,x,y,flags,param):
	if (event == cv2.EVENT_LBUTTONDOWN and x>430 and x<530 and y>400 and y<455): #captura para documento
		rostro= False
		fotos=0		
		while (rostro == False):
			ret, foto1 =video_capture2.read()
			re_foto1= cv2.resize (foto1, (300, 300))
			faces, eyes=detect_face(re_foto1)
			fotos+=1
			if(fotos>50):
				break
			if (faces is not None):
				rostro = True
				draw_rectangle(re_foto1, faces[0])
				break
		re_fondo[320:620, 5:305]=re_foto1

	if (event == cv2.EVENT_LBUTTONDOWN and x>430 and x<530 and y>470 and y<520): #captura para camara
		ret, foto2 =video_capture1.read()
		re_foto2= cv2.resize (foto2, (300, 300))
		faces, eyes=detect_face(re_foto2)
		if (faces is not None):
			draw_rectangle(re_foto2, faces[0])
		if (eyes is not None):
			for eye in eyes:
				draw_rectangle(re_foto2, eye)
		re_fondo[5:305, 5:305]=re_foto2
	
	elif (event == cv2.EVENT_LBUTTONDOWN and x>430 and x<530 and y>540 and y<590):
		re_fondo[5:305,5:305]=fondo_copy[5:305,5:305]
		re_fondo[320:620, 5:305]=fondo_copy[320:620, 5:305]
		


def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5);
	eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6);
	if(len(faces)==0):
		return None, None
	elif (len(eyes)==0):
		return faces, None	
	return faces,eyes

def draw_rectangle(img, rect,):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (1,220,255), 2)
		
img=re_fondo
cv2.namedWindow('img')
cv2.setMouseCallback('img',button_actions)


re_fondo[350:400, 390:520]=logo
#botones y textos

re_fondo[400:460, 430:530]=boton
cv2.putText(re_fondo, "Documento", (435,432), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 0)

re_fondo[468:528, 430:530]=boton
cv2.putText(re_fondo, "Foto", (445,505), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 0)

re_fondo[538:598, 430:530]=boton
cv2.putText(re_fondo, "limpiar", (440,570), cv2.FONT_HERSHEY_DUPLEX, 0.53, (0, 0, 0), 0)


while True:
	ret2, foto1= video_capture2.read()
	ret, live = video_capture1.read()
	re_live= cv2.resize (live, (300, 300))
	faces, eyes = detect_face(re_live)
	if (faces is not None):
		draw_rectangle(re_live, faces[0])
	re_fondo[5:305, 315:615]=re_live
	
	cv2.imshow('img',re_fondo )  
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture1.release()
cv2.destroyAllWindows()	

#Miguel Perez - migaperezber@hotmail.com
