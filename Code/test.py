import numpy as np
import random

train_ratio = 0.95

random.seed(1)

conversion_labels = {'republican' : 0, 'democrat' : 1, 'n' : 0, 'y' : 1, '?' : 2}
f = open('datasets/house-votes-84.data', 'r')

data = np.genfromtxt(f, delimiter=",", dtype=None)

attributes = []
classes = []
for i in range(len(data)):
    line = data[i]
    attr = []
    classes.append(conversion_labels[line[0]])
    for j in range(len(line) - 1):
        attr.append(conversion_labels[line[j + 1]])
    attributes.append(attr)

random.shuffle(data)

number_of_trains = int(train_ratio * len(data));
number_of_tests = len(data) - number_of_trains;

train = attributes[:number_of_trains]
train_labels = classes[:number_of_trains]

test = attributes[:number_of_tests]
test = classes[:number_of_tests]

print train

