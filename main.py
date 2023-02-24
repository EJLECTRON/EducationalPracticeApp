from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication
from sys import argv, exit
from cv2 import COLOR_BGR2RGB


from ui_mainInterface import Ui_MainWindow
from videoThread import *
from imageProcessingModule import *


def numOfPorts():
    """Test the ports and returns the last working port. It works if working
    ports are sequential"""
    is_working = True
    dev_port = 0

    while is_working:
        camera = VideoCapture(dev_port)

        if not camera.isOpened():
            is_working = False
            print("Port {0} is not working.".format(dev_port - 1))
        else:
            dev_port += 1

    return dev_port - 1


class MainWindow(QMainWindow):
    """ Main class """
    def __init__(self):
        super().__init__()

        #needed size of video image
        self.camera_width = 400
        self.camera_height = 300

        #initialization of main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.__functionallyInitialization()

        #select defalt camera
        self.__selectCamera(self.cameraIndex)

    def __functionallyInitialization(self):
        """ Implements all functionality and important flags"""
        self.isTypeOfSignalAutomatic = True
        self.isLightOn = True
        self.cameraIndex = 1

        self.__createBrightnessCalculationProcess()
        self.__buttonsActions()
        self.__comboBoxActions()

    def __createBrightnessCalculationProcess(self):
        self.brightnessCalculation = BrightnessCalculation()

    def __buttonsActions(self):
        self.ui.manualPushButton.clicked.connect(self.__manualControlling)
        self.ui.autoPushButton.clicked.connect(self.__automaticControlling)
        self.ui.typeSignalPushButton.clicked.connect(self.changeTypeSignal)
        self.ui.areaPushButton.clicked.connect(self.__switchCamera)

    def __comboBoxActions(self):
        """ Searching for available cameras and add text to comboboxes"""
        self.ui.typeSignalComboBox.addItem("Ввімнути світло")
        self.ui.typeSignalComboBox.addItem("Вимкнути світло")

        numOfCameras = numOfPorts()

        for counter in range(numOfCameras):
            self.ui.areaComboBox.addItem("Камера {0}".format(counter + 1))

    def __selectCamera(self, index):
        """ Reinitialize camera """
        # resize cameraLabel to needed size of video image
        self.ui.cameraLabel.resize(self.camera_width, self.camera_height)

        # create the video capture thread
        self.thread = VideoThread(index)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        """Calculates brightness and updates the image_label with a new opencv image"""

        isLightNeeded = self.brightnessCalculation.findOutDayOrNight(frame)

        if isLightNeeded:
            self.ui.stateDynamicLabel.setText("Подається сигнал")
        else:
            self.ui.stateDynamicLabel.setText("Сигнал не подається")

        qt_img = self.convert_cv_qt(frame)
        self.ui.cameraLabel.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cvtColor(frame, COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled = convert_to_Qt_format.scaled(self.camera_width, self.camera_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(scaled)

#----------------button actions---------------
    def changeTypeSignal(self):
        if not self.isTypeOfSignalAutomatic:
            match self.ui.typeSignalComboBox.currentIndex():
                case 0:
                    self.__turnOnLight(self.cameraIndex)
                case 1:
                    self.__turnOffLight(self.cameraIndex)
                case _:
                    Exception("Trouble with index of typeSignalComboBox")

    def __turnOnLight(self, indexOfArea):
        """ function that works if light is needed"""
        if not self.isLightOn:
            print('Light!')
            self.isLightOn = True
        else:
            print('Light has already been there')

    def __turnOffLight(self, indexOfArea):
        """ function that works if light isn't needed"""
        if self.isLightOn:
            print('Darkness!')
            self.isLightOn = False
        else:
            print('Darkness has already been there')

    def __switchCamera(self):
        """ Switch camera and change label"""
        if self.ui.areaComboBox.currentIndex() != self.cameraIndex:
            self.__selectCamera(self.ui.areaComboBox.currentIndex())
            self.cameraIndex = self.ui.areaComboBox.currentIndex()
            self.ui.cameraIndexLabel.setText("Камера {0}".format(self.cameraIndex + 1))

    def __manualControlling(self):
        if self.isTypeOfSignalAutomatic:
            self.ui.typeControlLabelDynamic.setText("Поточний тип керування: Ручний")
            self.ui.userCautionLabel.setText("")

            self.isTypeOfSignalAutomatic = False
        print("Manual control")

    def __automaticControlling(self):
        if not self.isTypeOfSignalAutomatic:
            self.ui.typeControlLabelDynamic.setText("Поточний тип керування: Автоматичний")
            self.ui.userCautionLabel.setText("Ви не можете змінювати параметри! Змініть тип керування.")

            self.isTypeOfSignalAutomatic = True
        print("Automatic control")



if __name__ == "__main__":
    app = QApplication(argv)

    mainWindow = MainWindow()
    mainWindow.show()

    exit(app.exec_())