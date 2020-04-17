from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from pygame import mixer
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

smile = 0
face_cascade = cv2.CascadeClassifier('/home/geethesh/Documents/imagevenv/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml') 
smile_cascade = cv2.CascadeClassifier('/home/geethesh/Documents/imagevenv/lib/python2.7/site-packages/cv2/data/haarcascade_smile.xml') 

class Smile:
	smile = 0 


sirippu = Smile()

def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
	sirippu.smile = smiles
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	C = dist.euclidean(eye[0], eye[3])
	
	ear = (A + B) / (2.0 * C)

	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,default="shape_predictor_68_face_landmarks.dat",help="path to facial landmark predictor")
args = vars(ap.parse_args())
 
MOUTH_AR_THRESH = 0.3
EYE_AR_THRESH = 0.25
EYE_AR_MILLIS = 400

COUNTER = 0
ALARM_ON = False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(lebStart,lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rebStart,rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream().start()
mixer.init()
mixer.music.load("screaming_ben.mp3")
closeFlag = False

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=650)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = detect(gray,frame)
	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		mouth = shape[mStart:mEnd]
		leftEyeBrow = shape[lebStart:lebEnd]
		rightEyeBrow = shape[rebStart:rebEnd]
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0
		ear1=round(ear,2)
 		stear =str(ear1)
		

		mouthHull = cv2.convexHull(mouth)
		leftEyeBrowHull = cv2.convexHull(leftEyeBrow)
		rightEyeBrowHull = cv2.convexHull(rightEyeBrow)
		leftEyeHull = cv2.convexHull(leftEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (150, 0, 150), 1)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [rightEyeHull], -1, (150, 0, 150), 1)
		print sirippu.smile
		if sirippu.smile == ():
			if ear < EYE_AR_THRESH:
				if  not closeFlag:
					stopTime = time.time()*1000.0 + EYE_AR_MILLIS
					closeFlag = True
			else:
				stopTime = time.time()*1000.0 +1000
				closeFlag = False
				print "DRIVER AWAKE"
				mixer.music.stop()
			
			if time.time()*1000.0 > stopTime:
				print "DRIVER FELL ASLEEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
				mixer.music.play()
				stopTime = time.time()*1000.0 + 2000
			


		cv2.putText(frame, "EAR: {:.2f}".format(ear), (325, 30),
			cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()






