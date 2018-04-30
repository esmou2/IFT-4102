import numpy as np
import copy
import NeuralNet
import load_datasets

# X = (hours sleeping, hours studying), y = score on test
X = ([2, 9], [1, 5], [3, 6])
y = np.array(([92], [86], [89]), dtype=float)

X = [[0,0,1], [0,1,1], [1,0,1], [1,1,1]]

y = [0, 1, 1, 0]

n= 1;

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(0.03)
train_votes, train_labels_votes, test_votes, test_labels_votes = load_datasets.load_congressional_dataset(0.02)
train_monks, train_labels_monks, test_monks, test_labels_monks = load_datasets.load_monks_dataset(n)

train = train_votes
labels = train_labels_votes

NN = NeuralNet.NeuralNet(1, 2, len(train[0]), 1)
for i in xrange(1000):
    NN.train(train, labels)

print "Actual Output: \n" + str(labels)
print "Predicted Output: \n" + str(NN.forward(train).T[0])

    # [0,1,0,1,1,1,0,0,0,0,0,0,1,1,2,1], 0))