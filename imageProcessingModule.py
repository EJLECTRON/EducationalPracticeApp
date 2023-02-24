from cv2 import cvtColor, GaussianBlur, threshold, COLOR_RGB2HSV, THRESH_BINARY, split
import numpy as np

class BrightnessCalculation:
    def findOutDayOrNight(self, frame):
        """ main function to use"""

        changedFrame = self.__changeFrame(frame)

        brightness = self.__calculateBrightnessUsingDistribution(changedFrame)

        if brightness >= 170:
            return 0
        else:
            return 1

    def __changeFrame(self, frame):
        """Image processing"""
        hsvImg = cvtColor(frame, COLOR_RGB2HSV)

        blurred = GaussianBlur(hsvImg, (9, 9), 0)

        thresh = threshold(blurred, 80, 255, THRESH_BINARY)[1]

        return thresh

    def __calculateBrightnessUsingDistribution(self, frame):
        h, s, v = split(frame)

        height, width = v.shape[0], v.shape[1]

        v = np.sum(v)

        avr = v / (height * width)

        return avr



