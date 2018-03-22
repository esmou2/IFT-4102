import numpy as np
import random

numero_dataset = 1

random.seed(1)

f_train = open('datasets/monks-' + `numero_dataset` + '.train', 'r')
f_test = open('datasets/monks-' + `numero_dataset` + '.test', 'r')


data_train = np.genfromtxt(f_train, delimiter=" ", dtype=str)
data_test = np.genfromtxt(f_test, delimiter=" ", dtype=str)

random.shuffle(data_train)

train = []
train_label = []
for i in range(len(data_train)):
    line = data_train[i]
    train.append(line[1:-1])
    train_label.append(line[-1])

random.shuffle(data_test)

test = []
test_label = []
for i in range(len(data_test)):
    line = data_test[i]
    test.append(line[1:-1])
    test_label.append(line[-1])

