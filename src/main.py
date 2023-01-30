import sys
import cv2
import numpy as np
from collections import defaultdict
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase
cred = credentials.Certificate("../credentials/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
	'storageBucket': 'smart-attendance-system-3a795.appspot.com'
})
db = firestore.client()

select_ui = uic.loadUiType("main.ui")[0]

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
        AttendanceSystem.setClass(self.cmb_class.currentText(), self.classDocs[self.cmb_class.currentText()])
        AttendanceSystem.show()
        SelectWindow.hide()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

class AttendanceSystem(QWidget):
    def __init__(self):
        super().__init__()
        global selectedClass
        global selectedClassInfo
        self.setWindowTitle("Smart Attendance System 1.0")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabelName = QLabel('Name: ')
        self.textLabelStatus = QLabel('Status: ')
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabelName)
        vbox.addWidget(self.textLabelStatus)
        
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def setClass(self, selectedClass, selectedClassInfo):
        self.selectedClass = selectedClass
        self.selectedClassInfo = selectedClassInfo

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