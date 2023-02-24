from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
from cv2 import VideoCapture, CAP_DSHOW

class VideoThread(QThread):
    """ Class that handles camera thread"""
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        super().__init__()

        self.run_flag = True

        #index of camera
        self.index = index

        # capture from selected camera
        self.cap = VideoCapture(self.index, CAP_DSHOW)

    def run(self):
        #get and sent frame to app
        while self.run_flag:
            self.ret, self.frame = self.cap.read()

            if self.ret:
                self.change_pixmap_signal.emit(self.frame)
            else:
                self.run_flag = False

        # shut down capture system
        super().quit()

    def stop(self):
        self.run_flag = False