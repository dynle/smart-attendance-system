# IDEA: 학습하기 누르면 collect한 다음 clf 모델을 다시 학습하고, 출석하기 누르면 훈련된 모델 사용해서 바로 face recognition

"""
	Uses KNN classifier which is supervised machine learning
"""

import pickle
import face_recognition
import cv2
import math
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, firestore, storage

##############################################
# set date and class name for test
day = 'Tue'
class_name = 'class4'
##############################################

font = cv2.FONT_HERSHEY_DUPLEX
now = datetime.now()
today = now.strftime("%Y-%m-%d")

# Firebase Initialization
cred = credentials.Certificate('src/credentials/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
	'storageBucket': 'smart-attendance-system-3a795.appspot.com'
})
db = firestore.client()

# Load the trained model from firebase storage
bucket = storage.bucket()
blob = bucket.blob("trained_knn_model.clf")
blob.download_to_filename("model_from_fb.clf")
with open("model_from_fb.clf","rb") as f:
    knn_clf = pickle.load(f)

# Get the student list and participants from firestore
class_ref = db.collection(day).document(class_name)
doc = class_ref.get()
if doc.exists:
	student_list = doc.to_dict()['students']
	remaining_students = student_list.copy()
else:
	print(u'No such document!')

date_ref = db.collection(day).document(class_name).collection('history').document(today)
doc = date_ref.get()
if doc.exists:
	remaining_students = [e for e in student_list if e not in doc.to_dict()['participants']]
else:
	db.collection(day).document(class_name).collection('history').document(today).set({u'participants': []})

# Set variables
face_locations = []
detected_list = []
result_flag = False

# https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage
# IDEA: Possible to use face_match_threshold = 0.2 when train the model with real people
def face_distance_to_conf(face_distance, face_match_threshold=0.2):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def predict(rgb_small_frame, model):
	"""
	Recognizes faces in given frame using a trained KNN classifier
	"""
	face_locations = face_recognition.face_locations(rgb_small_frame)

	# If no faces are found in the image, return an empty result.
	if len(face_locations) == 0:
		global detected_list
		detected_list = []
		return []

	# find encodings for faces in the frame
	faces_encodings = face_recognition.face_encodings(rgb_small_frame,known_face_locations=face_locations)

	# use the KNN model to find the best matches for the frame face
	closest_distances = model.kneighbors(faces_encodings,n_neighbors=5)
	face_distance = [closest_distances[0][i][0] for i in range(len(face_locations))][0]
	print("face_distance: ",face_distance)
	accuracy = face_distance_to_conf(face_distance)
	print("acc: ",accuracy)
	# IDEA: Possible to use distance_threshold = 0.2 when train the model with real people
	# Using a lower threshold than 0.6 makes the face comparison more strict.
	distance_threshold = 0.2
	are_matches = [closest_distances[0][i][0] <=
				distance_threshold for i in range(len(face_locations))]

	# predict classes and remove classification that aren't within the threshold
	return [(pred, loc, acc) if rec else ("Unknown", loc, acc) for pred, loc, rec, acc in zip(model.predict(faces_encodings), face_locations, are_matches, [accuracy])]

def show_labels(frame, name, top, right, bottom ,left, rec_color, acc=None):
	top *= 4
	right *= 4
	bottom *= 4
	left *= 4

	# Draw a box around the face
	cv2.rectangle(frame,(left-20,top-20),(right+20,bottom+20),(0,255,0),2)

	# Draw a label with a name below the face
	if name in remaining_students:
		cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), rec_color, cv2.FILLED)
		cv2.putText(frame, f"{name} {round(acc,2)}", (left -20, bottom + 15), font, 1, (255, 255, 255), 2)
	elif name == "Unknown":
		cv2.rectangle(frame, (left-20, bottom -15), (right+20, bottom+20), rec_color, cv2.FILLED)
		cv2.putText(frame, name, (left -20, bottom + 15), font, 1, (255, 255, 255), 2)

def take_attendance(name):
	remaining_students.remove(name)
	print("left remaining_students: ",remaining_students)
	current_time = now.strftime("%H:%M:%S")
	print(f'{name} attended: {current_time}')

	# Update the participants
	date_ref.update({u'participants': firestore.ArrayUnion([name])})





if __name__ == "__main__":
	video_capture = cv2.VideoCapture(0)

	frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

	if not video_capture.isOpened():
		print('Failed to open camera')
	else:
		while True:
			ret,frame = video_capture.read()
			# Resize frame of video to 1/4 size for faster face recognition processing
			small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
			# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
			rgb_small_frame = small_frame[:,:,::-1]

			# Show a message for successful attendance 
			if result_flag:
				if (result_end_time - datetime.now()).total_seconds() > 0:
					cv2.rectangle(frame, (int(frame_width/4), int(frame_height/10)), (int(frame_width*3/4), int((frame_height/10)*2)), (111, 191, 2), cv2.FILLED)
					cv2.putText(frame, name, (int(frame_width/4) + 15, int(frame_height/10) + 35), font, 1, (255, 255, 255), 2)
					cv2.putText(frame, "Successfully Attended", (int(frame_width/4) + 15, int(frame_height/10) + 65), font, 1, (255, 255, 255), 1)
				else:
					result_flag = False

			predictions = predict(rgb_small_frame,model=knn_clf)

			for name, (top, right, bottom, left), acc in predictions:
				print("- Found {} at ({}, {})".format(name, left, top))
				print(detected_list)

				# Display results overlaid on frame in real-time
				show_labels(frame, name, top, right, bottom, left, (255, 0, 0), acc)
				
				# Save the result to detected_list and initialize the list if the result is different from the results in the list
				if name in remaining_students:
					# 최소 30프레임? loop? 동안 같은 사람이면 본인인정 → 출석
					if len(detected_list) == 30:
						take_attendance(name)

						result_flag = True
						result_end_time = datetime.now() + timedelta(seconds=3)
						
						# IDEA: Add the name on list
						cv2.rectangle(frame, (left*4-20, bottom*4 -15), (right*4+20, bottom*4+20), (0,255,0), cv2.FILLED)
						cv2.putText(frame, "Attended", (left*4 -20, bottom*4 + 15), font, 1, (255, 255, 255), 2)

						detected_list = []
					elif len(detected_list) == 0 or name == detected_list[-1]:
						detected_list.append(name)
					elif name != detected_list[-1]:
						detected_list = []
				elif name == "Unknown":
					show_labels(frame, "Unknown", top, right, bottom, left, (0, 0, 255))
					detected_list = []

			cv2.imshow("attendance system",frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break



		video_capture.release()
		cv2.destroyAllWindows()
		# f.close()