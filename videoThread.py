from PyQt5.QtCore import pyqtSignal, QThread
import numpy as np
from cv2 import VideoCapture

class VideoThread(QThread):
    """ Class that handles camera thread"""
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        super().__init__()

        self.run_flag = True

        #index of camera
        self.index = index

    def run(self):
        # capture from selected camera
        cap = VideoCapture(self.index)

        #get and sent frame to app
        while self.run_flag:
            self.ret, self.frame = cap.read()

            if self.ret:
                self.change_pixmap_signal.emit(self.frame)

        # shut down capture system
        super().quit()