from scipy.spatial import distance as dt
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np

mixer.init()
mixer.music.load("alarm.wav")

def eye_aspect_ratio(eye):
	A = dt.euclidean(eye[1],eye[5])
	B = dt.euclidean(eye[2],eye[4])
	C = dt.euclidean(eye[0],eye[3])
	ear =(A + B)/(2.0 * C)
	return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

thresh = 0.25
frame_check = 20
yawn_thresh= 15
yawn_frame_check = 30
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\dhanu\Downloads\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

#To access camera
cap=cv2.VideoCapture(0)
flag=0
yawn_flag=0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	for i in subjects:
		shape = predict(gray, i)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		distance = lip_distance(shape)
		lip = shape[48:60]

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		#To display face marking
		cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
		cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

		cv2.drawContours(frame,[lip],-1,(0,255,0),1)

		if ear < thresh:
			flag += 1
			#To display eye movement
			#print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
		else:
			flag = 0

		if(distance > yawn_thresh):
                    yawn_flag += 1
                   #print(yawn_flag)
                    if yawn_flag >= yawn_frame_check:
                        cv2.putText(frame, "*************YAWN ALERT!*************", (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "*************YAWN ALERT!*************", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
		else:
			yawn_flag=0

	#To display fetched camera data shows like a small frame	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release()