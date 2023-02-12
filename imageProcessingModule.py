import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

class FrameProcessing:
    def __init__(self):
        pass

    def figureOutDayOrNight(self, frame):
        changedFrame = self.changeFrame(frame)
        
        brightness = self.croppingImgAndCalculateBrightness(changedFrame, 4)

        brightness_1 = self.calculateBrightnessUsingDistribution(frame)

        brightness_1 = self.__normalDistribution(brightness_1, np.mean(brightness_1), np.std(brightness_1))

        print(str(brightness) + "---------" + str(np.mean(brightness_1)))

        return [brightness, np.mean(brightness_1)]
        

    def resume(self):
        pass

    def stop(self):
        pass

    def changeFrame(self, frame):
        hsvImg = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        blurred = cv2.GaussianBlur(hsvImg, (9, 9), 0)

        thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.flip(thresh, 1)

        return frame

    def croppingImgAndCalculateBrightness(self, frame, divider):
        width, height, brightness = frame.shape[1], frame.shape[0], 0

        # 1/divider part of Frame
        deltaX, deltaY = width // divider, height // divider
        
        for y in range(0, height, deltaY):
            for x in range(0, width, deltaX):
                if (height - y) < deltaY and (width - x) < deltaX:
                    tempFrame = frame[y:height, x:width]
                elif (height - y) < deltaY:
                    tempFrame = frame[y:height, x:x + deltaX]
                elif (width - x) < deltaX:
                    tempFrame = frame[y:y + deltaY, x:width]
                else:
                    tempFrame = frame[y:y+deltaY, x:x+deltaX]

                brightness += self.__calculateFrameBrightness(tempFrame)

        return brightness / pow(divider, 2)

    def calculateBrightnessUsingDistribution(self, frame):
        h, s, v = cv2.split(frame)

        return v

    def __normalDistribution(self, x, mean, sd):
        prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
        return prob_density
                
    def __calculateFrameBrightness(self, frame):
        if len(frame.shape) == 3:
            return np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(frame)


if __name__ == "__main__":
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    frameProcessing = FrameProcessing()

    croppingData = np.array([])
    distributionData = np.array([])

    for i in range(350):

        ret, frame = camera.read()

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        brightness = frameProcessing.figureOutDayOrNight(frame)

        croppingData = numpy.append(croppingData, brightness[0])
        distributionData = numpy.append(distributionData, brightness[1])

    if len(croppingData) == len(distributionData):
        x = list(range(len(croppingData)))

        plt.plot(x, croppingData)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title("Cropping data")
        plt.show()

        plt.plot(x, distributionData)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title("Distribution data")
        plt.show()




