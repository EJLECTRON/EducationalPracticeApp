import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

class BrightnessCalculation:
    def findOutDayOrNight(self, frame):
        """ main function to use"""

        changedFrame = self.__changeFrame(frame)

        brightness = self.__calculateBrightnessUsingDistribution(changedFrame)

        if brightness >= 170:
            return 0
        else:
            return 1
    
    def __comparisonOfTwoMethods(self, frame):
        """ test of different methods"""
        changedFrame = self.__changeFrame(frame)
        
        brightness = self.__croppingImgAndCalculateBrightness(changedFrame, 4)

        brightness_1 = self.__calculateBrightnessUsingDistribution(changedFrame)

        print(str(brightness) + "---------" + str(np.mean(brightness_1)))

        return [brightness, np.mean(brightness_1)]

    def __changeFrame(self, frame):
        """Image processing"""
        hsvImg = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        blurred = cv2.GaussianBlur(hsvImg, (9, 9), 0)

        thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]

        return thresh

    def __croppingImgAndCalculateBrightness(self, frame, divider):
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

    def __calculateBrightnessUsingDistribution(self, frame):
        h, s, v = cv2.split(frame)

        height, width = v.shape[0], v.shape[1]

        v = np.sum(v)

        avr = v / (height * width)

        return avr
                
    def __calculateFrameBrightness(self, frame):
        if len(frame.shape) == 3:
            return np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(frame)

def oldTesting():
    brightnessCalculation = BrightnessCalculation()
    temp_res_y, temp_res_x = [], []

    test_data_list = load_images_from_folder("K:\\Programming\\educational practice\\images\\training_data")

    for sensitivity in range(100, 200, 5):

        result = []

        print("Start of testing ^_^")

        for image, target in test_data_list:
            tempResult = brightnessCalculation.findOutDayOrNight(image, sensitivity)

            if tempResult == target:
                result.append(1)
            else:
                result.append(0)

        print("End of testing ^_^")

        result = np.asarray(result)

        accuracy = result.sum() / result.size

        temp_res_x.append(sensitivity)
        temp_res_y.append(accuracy)

    temp_res_y = np.asarray(temp_res_y)
    temp_res_x = np.asarray(temp_res_x)

    max = np.amax(temp_res_y)
    index = np.where(temp_res_y == max)

    print(max, temp_res_x[index[0]])

    plt.plot(temp_res_x, temp_res_y)
    plt.show()

def load_images_from_folder(folder):
    """ Loads images for OldTesting"""
    images = []
    print("Data is loading, please wait...")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))

            if "day" in filename:
                images.append((img, 0))
            elif "night" in filename:
                images.append((img, 1))

    print("Data has loaded!")
    return images



if __name__ == "__main__":
    pass




