import cv2
import os
import numpy as np
from sklearn import preprocessing
from scipy.special import expit
from time import time

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # amount of input, hidden and output nodes
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learning rate
        self.lr = learning_rate

        # matrix of coupling coefficients between input and hidden nodes
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        # matrix of coupling coefficients between hidden and output nodes
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = expit(hidden_inputs)


        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = expit(final_inputs)


        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)


        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = expit(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = expit(final_inputs)

        return final_outputs


def load_images_from_folder(folder):
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

def training_and_using_nn_with_manual_normalization(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate, training_data_list, test_data_list):

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("Start of training...")

    for e in range(epochs):
        start = time()
        for data in training_data_list:
            img, flag = data[0], data[1]

            hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            v = hsvImg[: , :, 2]

            inputs = v / 255.0 * 0.99 + 0.01

            targets = np.zeros(output_nodes) + 0.01

            targets[flag] = 0.99

            nn.train(inputs, targets)

        end = time()

        print(f"Time of epoch {e + 1} = {end - start}")

    print("End of training!")

    print("Start of testing ^_^")

    score_list = []

    for data in test_data_list:
        img, correct_value = data[0], data[1]

        hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        v = hsvImg[: , :, 2]

        inputs = v / 255.0 * 0.99 + 0.01

        outputs = nn.query(inputs)

        value = np.argmax(outputs)

        if value == correct_value:
            score_list.append(1)
        else:
            score_list.append(0)

    print("End of testing ^_^")

    score_array = np.asarray(score_list)

    accuracy = score_array.sum() / score_array.size

    data_of_efficiency.append(accuracy)

    print(data_of_efficiency)

def training_and_using_nn_with_scikit_normalization(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate, training_data_list, test_data_list):

    NN = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("Start of training...")

    for e in range(epochs):
        start = time()
        for data in training_data_list:
            img, flag = data[0], data[1]

            hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            v = hsvImg[: , :, 2]

            inputs = preprocessing.normalize(v)

            targets = np.zeros(output_nodes) + 0.01

            targets[flag] = 0.99

            NN.train(inputs, targets)

        end = time()

        print(f"Time of epoch {e + 1} = {end - start}")

    print("End of training!")

    print("Start of testing ^_^")

    score_list = []

    for data in test_data_list:
        img, correct_value = data[0], data[1]

        hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        v = hsvImg[: , :, 2]

        inputs = preprocessing.normalize(v)

        outputs = NN.query(inputs)

        value = np.argmax(outputs)

        if value == correct_value:
            score_list.append(1)
        else:
            score_list.append(0)

    print("End of testing ^_^")

    score_array = np.asarray(score_list)

    accuracy = score_array.sum() / score_array.size

    data_of_efficiency.append(accuracy)

    print(data_of_efficiency)

if __name__ == "__main__":
    input_nodes = 128
    hidden_nodes = 65
    output_nodes = 2

    training_data_list = load_images_from_folder("K:\\Programming\\educational practice\\images\\training_data")

    folder = "K:\\Programming\\educational practice\\images\\test_data"

    test_data_list = load_images_from_folder(folder)

    data_of_efficiency = []

    learning_rate = 0.1

    epochs = 30

    training_and_using_nn_with_scikit_normalization(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate, training_data_list, test_data_list)

    training_and_using_nn_with_manual_normalization(epochs, input_nodes, hidden_nodes, output_nodes, learning_rate, training_data_list, test_data_list)