# TODO: 아직까지 SK를 DY라고 판단하는 현상 → 학습데이터 늘리고 모델 학습
# TODO: 만약 firebase 데이터에 변화가 있으면 clf를 다시 훈련하고, 없으면 있던 데이터 그대로 씀

"""
	Uses KNN classifier which is supervised machine learning
	test
"""

import pickle
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

with open("trained_knn_model.clf","rb") as f:
    knn_clf = pickle.load(f)

video_capture = cv2.VideoCapture(0)

known_faces_names = [
	"DY",
	"SK",
	"BIden"
]

# TODO: students list는 firebase에서 가져오기
students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


# Use firestore instead of saving txt file
f = open(current_date+'.txt','w+')

def predict(rgb_small_frame, model):
	"""
	Recognizes faces in given frame using a trained KNN classifier
	"""

	face_locations = face_recognition.face_locations(rgb_small_frame)

	# If no faces are found in the image, return an empty result.
	if len(face_locations) == 0:
		return []

	# find encodings for faces in the frame
	faces_encodings = face_recognition.face_encodings(rgb_small_frame,known_face_locations=face_locations)

	# use the KNN model to find the best matches for the frame face
	closest_distances = model.kneighbors(faces_encodings,n_neighbors=1)
	distance_threshold = 0.5
	are_matches = [closest_distances[0][i][0] <=
				distance_threshold for i in range(len(face_locations))]

	# predict classes and remove classification that aren't within the threshold
	return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(model.predict(faces_encodings), face_locations, are_matches)]

def show_labels(frame, predictions):
	for name, (top, right, bottom ,left) in predictions:
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4
		
		# Draw a box around the face
		cv2.rectangle(frame,(left-20,top-20),(right+20,bottom+20),(0,255,0),2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left -20, bottom + 15), font, 1, (255, 255, 255), 2)

		if name in students:
			students.remove(name)
			print("left students: ",students)
			current_time = now.strftime("%H:%M:%S")
				# Save it to the firestore
			f.write(f'{name} {current_time}\n')
			print(f'{name} attended: {current_time}')


if not video_capture.isOpened():
	print('Failed to open camera')
else:
	while True:
		_,frame = video_capture.read()
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:,:,::-1]

		predictions = predict(rgb_small_frame,model=knn_clf)

		# Print results on the console
		for name, (top, right, bottom, left) in predictions:
			print("- Found {} at ({}, {})".format(name, left, top))
		
		# Display results overlaid on frame in real-time
		show_labels(frame, predictions)

		cv2.imshow("attendance system",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
	f.close()