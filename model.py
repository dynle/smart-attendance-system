import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)


# List of students (Get photos from firebase and do this in for loop)
DY_image = face_recognition.load_image_file("photos/DY.jpeg")
DY_encoding = face_recognition.face_encodings(DY_image)[0]

SK_image = face_recognition.load_image_file("photos/SK.jpeg")
SK_encoding = face_recognition.face_encodings(SK_image)[0]

known_face_encoding = [
	DY_encoding,
	SK_encoding
]

known_faces_names = [
	"DY",
	"SK"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
# process_this_frame = True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


# Use firestore instead of saving txt file
f = open(current_date+'.txt','w+')

if not video_capture.isOpened():
	print('Failed to open camera')
else:
	while True:
		_,frame = video_capture.read()
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:,:,::-1]

# 		if process_this_frame:
		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
		
		face_names = []
		for face_encoding in face_encodings:
			matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
			name = ""
			face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
			best_match_index = np.argmin(face_distance)
			if matches[best_match_index]:
				name = known_faces_names[best_match_index]
			face_names.append(name)
			
			if name in known_faces_names:
				for (top,right,bottom,left) in face_locations:
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


		cv2.imshow("attendance system",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
	f.close()