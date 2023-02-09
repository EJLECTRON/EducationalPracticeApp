from PyQt5.QtCore import pyqtSignal, QThread
from threading import Thread
from imageProcessingModule import *

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, index, imageProcessing):
        super().__init__()

        self.run_flag = True
        self.index = index
        self.imageProcessing = imageProcessing

    def run(self):
        # capture from selected camera
        cap = cv2.VideoCapture(self.index)

        isImageProcessingProcessStarted = False

        while self.run_flag:
            self.ret, self.frame = cap.read()

            print(type(self.frame))

            if not self.imageProcessing.imageProcessingThread.is_alive():
                try:
                    self.imageProcessing.imageProcessingThread.start()
                except:
                    self.imageProcessing.imageProcessingThread = Thread(target=self.imageProcessing.figureOutDayOrNight, args=(self.imageProcessing, self.frame,))
                    self.imageProcessing.imageProcessingThread.start()

            if self.ret:
                self.change_pixmap_signal.emit(self.frame)

        self.imageProcessing.imageProcessingThread.join()

        # shut down capture system
        super().quit()