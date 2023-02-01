from datetime import datetime, timedelta
import math
import pickle
import sys
import cv2
import numpy as np
from collections import defaultdict
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
import firebase_admin
from firebase_admin import credentials, firestore, storage
import face_recognition

font = cv2.FONT_HERSHEY_DUPLEX
now = datetime.now()
today = now.strftime("%Y-%m-%d")

# Firebase Initialization
cred = credentials.Certificate('./credentials/serviceAccountKey.json')
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

select_ui = uic.loadUiType("./main.ui")[0]


class SelectWindow(QMainWindow, select_ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # Events for selecting the day
        self.cmb_day.currentIndexChanged.connect(self.daySelected)
        # Events for clicking the start btn
        self.btn_start.clicked.connect(self.startAttendanceSystem)
        self.daySelected()

    def daySelected(self):
        # Get the all classes of the selected day from firestore
        doc_ref = db.collection(self.cmb_day.currentText()[:3])
        doc = doc_ref.get()
        self.addClass(doc)
    
    def addClass(self, docs):
        self.cmb_class.clear()
        self.classDocs = defaultdict(dict)
        for doc in docs:
            self.cmb_class.addItem(doc.id)
            self.classDocs[doc.id] = doc.to_dict()
    
    def startAttendanceSystem(self):
        # Close the current window and open the attendance window
        if self.cmb_class.currentText() == "":
            QMessageBox.about(self, "Error", "Please select the class")
            return
        AttendanceSystem.setClass(self.cmb_day.currentText(), self.cmb_class.currentText(), self.classDocs[self.cmb_class.currentText()])
        AttendanceSystem.show()
        SelectWindow.hide()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, day, class_name, student_list, labelName, labelStatus):
        super().__init__()
        self.detected_list = []
        self.face_locations = []
        self.student_list = student_list
        self.remaining_students = student_list.copy()
        self.labelName = labelName
        self.labelStatus = labelStatus
        self.participants_list = []

        self.date_ref = db.collection(day).document(class_name).collection('history').document(today)
        doc = self.date_ref.get()

        if doc.exists:
            self.participants_list = doc.to_dict()['participants']
            self.remaining_students = [e for e in student_list if e not in self.participants_list]
        else:
            db.collection(day).document(class_name).collection('history').document(today).set({u'participants': []})
    
    def run(self):
        # capture from web cam
        video_capture = cv2.VideoCapture(0)

        frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        result_flag = False

        while True:
            ret,cv_img = video_capture.read()
            # Resize cv_img of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(cv_img,(0,0),fx=0.25,fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:,:,::-1]

            # Show a message for successful attendance 
            if result_flag:
                if (result_end_time - datetime.now()).total_seconds() > 0:
                    cv2.rectangle(cv_img, (int(frame_width/4), int(frame_height/10)), (int(frame_width*3/4), int((frame_height/10)*2)), (111, 191, 2), cv2.FILLED)
                    cv2.putText(cv_img, name, (int(frame_width/4) + 15, int(frame_height/10) + 35), font, 1, (255, 255, 255), 2)
                    cv2.putText(cv_img, "Successfully Attended", (int(frame_width/4) + 15, int(frame_height/10) + 65), font, 1, (255, 255, 255), 1)
                else:
                    result_flag = False

            predictions = self.predict(rgb_small_frame,model=knn_clf)

            for name, (top, right, bottom, left), acc in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                print(self.detected_list)
                self.labelName.setText(f"Name: {name}")
                if name in self.participants_list:
                    self.labelStatus.setText("Status: Attended")
                elif name not in self.student_list:
                    self.labelStatus.setText("Status: You are not taking this class!")
                else:
                    self.labelStatus.setText("Status: ")
                

                # Display results overlaid on cv_img in real-time
                self.show_labels(cv_img, name, top, right, bottom, left, (255, 0, 0), acc)

                # Save the result to detected_list and initialize the list if the result is different from the results in the list
                if name in self.remaining_students:
                    # 최소 30프레임? loop? 동안 같은 사람이면 본인인정 → 출석
                    if len(self.detected_list) == 30:
                        self.take_attendance(name)

                        result_flag = True
                        result_end_time = datetime.now() + timedelta(seconds=3)

                        cv2.rectangle(cv_img, (left*4-20, bottom*4 -15), (right*4+20, bottom*4+20), (0,255,0), cv2.FILLED)
                        cv2.putText(cv_img, "Attended", (left*4 -20, bottom*4 + 15), font, 1, (255, 255, 255), 2)

                        self.detected_list = []
                    elif len(self.detected_list) == 0 or name == self.detected_list[-1]:
                        self.detected_list.append(name)
                    elif name != self.detected_list[-1]:
                        self.detected_list = []
                elif name == "Unknown":
                    self.show_labels(cv_img, "Unknown", top, right, bottom, left, (0, 0, 255))
                    self.detected_list = []

            if ret:
                self.change_pixmap_signal.emit(cv_img)
    
    def predict(self, rgb_small_frame, model):
        """
        Recognizes faces in given frame using a trained KNN classifier
        """
        self.face_locations = face_recognition.face_locations(rgb_small_frame)

        # If no faces are found in the image, return an empty result.
        if len(self.face_locations) == 0:
            self.detected_list = []
            return []

        # find encodings for faces in the cv_img
        faces_encodings = face_recognition.face_encodings(rgb_small_frame,known_face_locations=self.face_locations)

        # use the KNN model to find the best matches for the cv_img face
        closest_distances = model.kneighbors(faces_encodings,n_neighbors=5)
        face_distance = [closest_distances[0][i][0] for i in range(len(self.face_locations))][0]
        accuracy = self.face_distance_to_conf(face_distance)
        print(f"acc: {int(round(accuracy, 3)*100)}%")
        print()
        # Using a lower threshold than 0.6 makes the face comparison more strict.
        distance_threshold = 0.3
        are_matches = [closest_distances[0][i][0] <=
                    distance_threshold for i in range(len(self.face_locations))]

        # predict classes and remove classification that aren't within the threshold
        return [(pred, loc, acc) if rec else ("Unknown", loc, acc) for pred, loc, rec, acc in zip(model.predict(faces_encodings), self.face_locations, are_matches, [accuracy])]

        # https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage
    def face_distance_to_conf(self, face_distance, face_match_threshold=0.3):
        if face_distance > face_match_threshold:
            range = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

    def show_labels(self, cv_img, name, top, right, bottom ,left, rec_color, acc=None):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(cv_img,(left-20,top-20),(right+20,bottom+20),(0,255,0),2)

        # Draw a label with a name below the face
        if name in self.remaining_students:
            cv2.rectangle(cv_img, (left-20, bottom -15), (right+20, bottom+20), rec_color, cv2.FILLED)
            cv2.putText(cv_img, f"{name} {round(acc,2)}", (left -20, bottom + 15), font, 1, (255, 255, 255), 2)
        elif name == "Unknown":
            cv2.rectangle(cv_img, (left-20, bottom -15), (right+20, bottom+20), rec_color, cv2.FILLED)
            cv2.putText(cv_img, name, (left -20, bottom + 15), font, 1, (255, 255, 255), 2)

    def take_attendance(self, name):
        self.remaining_students.remove(name)
        current_time = now.strftime("%H:%M:%S")

        # Update the participants
        self.date_ref.update({u'participants': firestore.ArrayUnion([name])})
        self.labelStatus.setText("Status: Attended")
        self.participants_list.append(name)

class AttendanceSystem(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Attendance System 1.0")
        self.disply_width = 800
        self.display_height = 600
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.classInfo = QLabel('')
        self.textLabelName = QLabel('Name: ')
        self.textLabelStatus = QLabel('Status: ')
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.classInfo)
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabelName)
        vbox.addWidget(self.textLabelStatus)
        
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)


    def setClass(self, selectedDay, selectedClass, selectedClassInfo):
        self.selectedDay = selectedDay
        self.selectedClass = selectedClass
        self.selectedClassInfo = selectedClassInfo
        self.classInfo.setText(selectedClass + ' (' + selectedDay + ')')

        # create the video capture thread
        self.thread = VideoThread(selectedDay[:3], selectedClass, selectedClassInfo['students'], self.textLabelName, self.textLabelStatus)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    SelectWindow = SelectWindow()
    AttendanceSystem = AttendanceSystem()
    SelectWindow.show()
    app.exec_()