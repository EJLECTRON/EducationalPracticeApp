import cv2
import os
import numpy as np
from scipy.special import expit

def load_images_from_folder(folder):
    images = []
    print("Training data is loading, please wait...")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))

            if "day" in filename:
                images.append((img, 0))
            elif "night" in filename:
                images.append((img, 1))

    print("Training data has loaded!")
    return images

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #amount of input, hidden and output nodes
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        #learning rate
        self.lr = learning_rate

        #matrix of coupling coefficients between input and hidden nodes
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))

        # matrix of coupling coefficients between hidden and output nodes
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #sigmoid
        self.activation_function = lambda x: expit(x)

    def train(self, inputs_list, targets_list):

        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)


        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)


        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs

if __name__ == "__main__":
    input_nodes = 128
    hidden_nodes = 50
    output_nodes = 2

    learning_rate = 0.3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_list = load_images_from_folder("K:\\Programming\\educational practice\\images\\training_data")

    img, flag = training_data_list[0]

    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(img)

    inputs = v / 255.0 * 0.99 + 0.01

    targets = np.zeros(output_nodes) + 0.01

    targets[flag] = 0.99

    print("Start of training...")

    nn.train(inputs, targets)

    print("End of training!")

    print("Start of testing ^_^")

    folder = "K:\\Programming\\educational practice\\images\\test_data"
    filename = "night_image (2974).jpg"

    img = cv2.imread(os.path.join(folder, filename))

    target = []

    if img is not None:
        img = cv2.resize(img, (128, 128))

        if "day" in filename:
            target = [img, 0]
        elif "night" in filename:
            target = [img, 1]

    hsvImg = cv2.cvtColor(target[0], cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(target[0])

    inputs = v / 255.0 * 0.99 + 0.01

    outputs = nn.query(inputs)

    label = np.argmax(outputs)

    if label == target[1]:
        print("(👍 ͡❛ ‿●‿ ͡❛)👍")

    print("End of testing ^_^")
