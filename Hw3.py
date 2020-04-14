import random
import gzip
import numpy
import pickle


#calculam eroarea pentru ultimul strat de neuroni.
#gradientul este transmis  straturilor de neuroni  si calculam updateul pentru parametrii


def load_db(file_name):
    with gzip.open(file_name, 'rb') as fin:
            train_set, valid_set, test_set = pickle.load(fin, encoding='latin1')
            return train_set, valid_set, test_set



def get_vector_from_digit(y):
    vector = numpy.zeros((10, 1))
    vector[y] = 1.0
    return vector


def format_data():
    train_set, valid_set, test_set = load_db(r'mnist.pkl.gz')

    # 1
    training_inputs = [numpy.reshape(x, (784, 1)) for x in train_set[0]]
    # digit-vector (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0]
    training_results = [get_vector_from_digit(y) for y in train_set[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [numpy.reshape(x, (784, 1)) for x in valid_set[0]]
    validation_data = zip(validation_inputs, valid_set[1])

    test_inputs = [numpy.reshape(x, (784, 1)) for x in test_set[0]]
    test_data = zip(test_inputs, test_set[1])

    return training_data, validation_data, test_data


def learn():
    global training_data, test_data

    learning_rate = 0.7
    lambda_ = 0.4 #regularization parameter
    number_of_iterations = 2
    mini_batch_size = 5
    n = 50000


    # biases-uri pentru al doilea layer si ultimul (distributie normala)
    biases = [numpy.random.randn(y, 1) for y in [100, 10]]
    # costurile pentru fiecare layer (distributie normala)
    # all values will be initialized with a random value from a normal distribution
    # with mean 0  and a standard deviation  sqrt(x) = total number of connections
    weights = [numpy.random.randn(y, x) / numpy.sqrt(x) for x, y in [(784, 100), (100, 10)]]

    for iteration in range(number_of_iterations):
        random.shuffle(training_data)

        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            nabla_b = [numpy.zeros(b.shape) for b in biases]  # [[100][1], [10][1]]
            # [[100][784], [10][100]] mini batch total weight gradient
            nabla_w = [numpy.zeros(w.shape) for w in weights]

            for x, y in mini_batch: #y este target
                bias_gradient, weight_gradient = backpropagation(x, y, biases, weights)  # gradients pentru un input
                # suma pentru mini-batch
                nabla_b = [nb + bg for nb, bg in zip(nabla_b, bias_gradient)]
                nabla_w = [nw + wg for nw, wg in zip(nabla_w, weight_gradient)]
            #L2 regularization slide 26
            weights = [
                (1 - learning_rate * (lambda_ / n)) * w -
                (learning_rate / len(mini_batch)) * nw for w, nw in zip(weights, nabla_w)
            ]
            biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(biases, nabla_b)]

        number_of_correct_outputs = test_accuracy(test_data, biases, weights)
        accuracy = float(number_of_correct_outputs) / 100

        print("Iteration {}: {} %".format(iteration + 1, accuracy))

# gradients for just one input
def backpropagation(x, y, biases, weights):
    # forward pass
    # activari
    z1 = x
    activation_1 = z1

    z2 = numpy.dot(weights[0], activation_1) + biases[0]  # input - ul fiecarui neuron de la layer 2
    # activarea fiecarui neuron de la layer 2, sigmoid(w_layer2 * activation_layer1 + bias_layer2) = z2
    activation_2 = sigmoid(z2)

    z3 = numpy.dot(weights[1], activation_2) + biases[1]  # inputul fiecarui neuron de la layer 3
    # activaticarea fiecarui neuron de la layer 3, softmax(w_layer3* activation_layer2 + bias_layer3) = z3
    activation_3 = softmax(z3)


    # eroare pentru layer 3 (target - output)
    delta_23 = activation_3 - y
    bias_gradient23 = delta_23
    # gradienti pentru costuri de la layer 2 la layer 3
    weight_gradient23 = numpy.dot(delta_23, activation_2.transpose())

    # eroare pentru layer 2
    # #delta_k, l+1 * w_ik, l+1
    delta_12 = numpy.dot(weights[1].transpose(), delta_23) * sigmoid_prime(z2)
    bias_gradient12 = delta_12
    # gradienti pentru greutati de la layer 1 la layer 2
    # cost function depends on a weight delta _i,l * y_k, l-1(activation of neuron k from the l-1 layer)
    weight_gradient12 = numpy.dot(delta_12, activation_1.transpose())

    bias_gradient = [bias_gradient12, bias_gradient23]
    weight_gradient = [weight_gradient12, weight_gradient23]

    return bias_gradient, weight_gradient  # gradient for the cost function C_x: delta_nabla_b, delta_nabla_w


def test_accuracy(test_data, biases, weights):
    number_of_correct_outputs = 0
    for x, y in test_data:
        output = find_output(x, biases, weights)
        digit = numpy.argmax(output)
        if digit == y:
            number_of_correct_outputs += 1

    return number_of_correct_outputs


def find_output(x, biases, weights):
    activation_2 = sigmoid(numpy.dot(weights[0], x) + biases[0])
    activation_3 = softmax(numpy.dot(weights[1], activation_2) + biases[1])

    return activation_3


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    exp_sum = sum(numpy.exp(zk) for zk in z)
    return numpy.exp(z) / exp_sum


if __name__ == '__main__':
    training_data, validation_data, test_data = format_data()
    training_data = list(training_data)
    test_data = list(test_data)
    validation_data = list(validation_data)
    learn()