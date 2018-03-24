import numpy as np
import random

def load_iris_dataset(train_ratio):

    random.seed(1)

    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

    f = open('datasets/bezdekIris.data', 'r')
    # TODO : le code ici pour lire le dataset
    number_of_trains = int(train_ratio * 150);

    data = np.genfromtxt(f, delimiter=",", dtype=str)

    np.random.shuffle(data)

    attributes = []
    classes = []
    for i in range(150):
        line = data[i]
        attr = []
        for j in range(4):
            attr.append(float(line[j]))
        attributes.append(attr)
        classes.append(conversion_labels[line[4]])

    train = attributes[:number_of_trains]
    train_labels = classes[:number_of_trains]

    test = attributes[number_of_trains:]
    test_labels = classes[number_of_trains:]

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)



def load_congressional_dataset(train_ratio):

    random.seed(1)
    conversion_labels = {'republican' : 0, 'democrat' : 1,
                         'n' : 0, 'y' : 1, '?' : 2}

    f = open('datasets/house-votes-84.data', 'r')

    # TODO : le code ici pour lire le dataset
    data = np.genfromtxt(f, delimiter=",", dtype=str)

    attributes = []
    classes = []
    for i in range(len(data)):
        line = data[i]
        attr = []
        classes.append(conversion_labels[line[0]])
        for j in range(len(line) - 1):
            attr.append(float(conversion_labels[line[j + 1]]))
        attributes.append(attr)

    np.random.shuffle(data)

    number_of_trains = int(train_ratio * len(data));

    train = attributes[:number_of_trains]
    train_labels = classes[:number_of_trains]

    test = attributes[number_of_trains:]
    test_labels = classes[number_of_trains:]

    return (train, train_labels, test, test_labels)


def load_monks_dataset(numero_dataset):

	# TODO : votre code ici, vous devez lire les fichiers .train et .test selon l'argument numero_dataset

    f_train = open('datasets/monks-' + `numero_dataset` + '.train', 'r')
    f_test = open('datasets/monks-' + `numero_dataset` + '.test', 'r')


    data_train = np.genfromtxt(f_train, delimiter=" ", dtype=str)
    data_test = np.genfromtxt(f_test, delimiter=" ", dtype=str)

    np.random.shuffle(data_train)

    train = []
    train_labels = []
    for i in range(len(data_train)):
        line = data_train[i]
        train.append([float(i) for i in line[1:-1]])
        train_labels.append(int(line[0]))

    np.random.shuffle(data_test)

    test = []
    test_labels = []
    for i in range(len(data_test)):
        line = data_test[i]
        test.append([float(i) for i in line[1:-1]])
        test_labels.append(int(line[0]))
    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)