from PyQt5.QtCore import pyqtSignal, QThread
from NeuralNetworkModule import *

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        super().__init__()

        self.run_flag = True
        self.index = index

    def run(self):
        # capture from selected camera
        cap = cv2.VideoCapture(self.index)

        while self.run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)

        # shut down capture system
        super().quit()