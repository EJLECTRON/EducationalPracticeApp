import cv2
import numpy as np
from matplotlib import pyplot as plt
from threading import Thread


class ImageProcessing:
    def __init__(self):
        self.neuralNetworkThread = Thread()

    def figureOutDayOrNight(self, frame):
        frame = self.changeFrame(frame)
        
        plt.imshow(frame)
        plt.show()

    def resume(self):
        pass

    def stop(self):
        pass

    def changeFrame(self, frame):
        hsvImg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(hsvImg, (9, 9), 0)

        thresh = cv2.threshold(blurred, 75, 215, cv2.THRESH_BINARY)[1]

        thresh = cv2.flip(thresh, 1)

        return thresh

    def showImg(self):
        pass

    def __cutImage(self, image, x, y):
        pass

    def __imageProcessing(self, image):
        pass


if __name__ == "__main__":
    camera = cv2.VideoCapture(1)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = camera.read()

        frame = ImageProcessing.changeFrame(ImageProcessing(),frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    camera.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
