import pickle, gzip
import numpy as np


def getData():
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set


def activation(input):
    if input > 0:
        input = 1
    else:
        input = 0
    return input


def targetCheck(t, digit):
    if t == digit:
        return 1
    else:
        return 0


def createAndTrainPerceptron(digit):
    weights = np.zeros(784)
    bias = 0
    nrIterations = 10
    while (nrIterations > 0):
        trainSet, validSet, testSet = getData()
        for iterator in range(len(trainSet[0])):
            x = trainSet[0][iterator]
            t = trainSet[1][iterator]
            z = np.dot(x, weights) + bias
            output = activation(z)
            error = targetCheck(t, digit) - output
            bias = bias + error * 0.4
            weights = weights + error * x * 0.4
        nrIterations -= 1
    return weights, bias


def test(listOfWeights, listOfBiases):
    goodResults = 0
    incorrect =0
    trainSet, validSet, testSet = getData()
    output = np.zeros(10)
    for iterator in range(len(testSet[0])):
        x = testSet[0][iterator]
        t = testSet[1][iterator]
        for digit in range(10):
            output[digit] = np.dot(x, listOfWeights[digit]) + listOfBiases[digit]
        if (output.argmax() == t):
            goodResults += 1
        else:
            incorrect += 1
    print("Correct instances : ", goodResults)
    print("Incorrect instances : ", incorrect)
    print("Precision :", goodResults / (goodResults+ incorrect))


def main():
    listOfWeights = np.zeros((10, 784))
    listOfBiases = np.zeros(10)

    for digit in range(10):
        listOfWeights[digit], listOfBiases[digit] = createAndTrainPerceptron(digit)

    test(listOfWeights, listOfBiases)


if __name__ == '__main__':
    main()