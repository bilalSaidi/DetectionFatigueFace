'''
        Mini Project Walid Zerman & Bilal saidi

        


'''

import cv2
import dlib
from scipy.spatial import distance

"""

the implementation EAR algorithme  from a paper 

LINK : https://www.semanticscholar.org/paper/Eye-Blink-Detection-Using-Facial-Landmarks-Soukupov%C3%A1-%C4%8Cech/cc029963bb86a2fad55eeb46f4da0d2a3962e6b7

"""
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio


# Start Video 

cap = cv2.VideoCapture(0)


hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        """
        The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures.
        These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset [https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/]
        The right eye using [36, 42].
        The left eye with [42, 48].
        """
        #  right eye
        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        #  left eye
        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR<0.20:
        	
        	cv2.putText(frame,"Are you Fatigue?",(20,400),
        		cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        	print("Drowsy")
        print(EAR)

    cv2.imshow("Emetions Fatigue", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()