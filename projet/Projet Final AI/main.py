import numpy as np
import sys
import load_datasets
import NeuralNet
import DecisionTree

from pylab import *

print "\n================= L O A D =================\n"

n = 1
k = 5.

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(0.9)
train_votes, train_labels_votes, test_votes, test_labels_votes = load_datasets.load_congressional_dataset(0.9)
train_monks, train_labels_monks, test_monks, test_labels_monks = load_datasets.load_monks_dataset(n)

# print "\n================= I N I T =================\n"
dt_iris = DecisionTree.DecisionTree()
dt_votes = DecisionTree.DecisionTree()
dt_monks = DecisionTree.DecisionTree()

nn_iris = NeuralNet.NeuralNet(1, 4, 4, 1)
nn_votes = NeuralNet.NeuralNet(1, 11, 16, 1)
nn_monks = NeuralNet.NeuralNet(1, 15, 6, 1)

# train

print "\n================= T R A I N =================\n"

print "\nDECISION TREE\n"

print "Bezdek Iris"
dt_iris.train(train_iris, train_labels_iris)

print "\nHouse votes"
dt_votes.train(train_votes, train_labels_votes)

print "\nMonks-{}".format(n)
dt_monks.train(train_monks, train_labels_monks)

print "\nNEURAL NETWORK\n"

print "Bezdek Iris"
for i in range(100):
    nn_iris.train(train_iris, train_labels_iris)

print "\nHouse votes"
for i in range(100):
    nn_votes.train(train_votes, train_labels_votes)

print "\nMonks-{}".format(n)
for i in range(100):
    nn_monks.train(train_monks, train_labels_monks)


# test

print "\n================= T E S T =================\n"

print "\nDECISION TREE\n"

print "Bezdek Iris"
dt_iris.test(test_iris, test_labels_iris)

print "\nHouse votes"
dt_votes.test(test_votes, test_labels_votes)

print "\nMonks-{}".format(n)
# dt_monks.test(train_monks, train_labels_monks)

print "\nNeural Network\n"

print "Bezdek Iris"
# nn_iris.test(test_iris, test_labels_iris)

# print "\nHouse votes"
# nn_votes.test(test_votes, test_labels_votes)

# print "\nMonks-{}".format(n)
# nn_monks.test(test_monks, test_labels_monks)

# #Nombre de neurones
# erreur_moyenne = []
# for i in range(4, 50):
#     nn_iris = NeuralNet.NeuralNet(3, i, 4, 1)
#     for j in range(int(k)):
#         nn_iris.train(train_iris, train_labels_iris)
#     erreur_moyenne.append(nn_iris.test(test_iris, test_labels_iris))

# plot(range(4, 50), erreur_moyenne)
# show()

# print "\nHouse votes"
# erreur_moyenne = []
# for i in range(4, 50):
#     nn_votes = NeuralNet.NeuralNet(3, i, 16, 1)
#     for j in range(int(k)):
#         nn_votes.train(train_votes, train_labels_votes)
#     erreur_moyenne.append(nn_votes.test(test_votes, test_labels_votes))

# plot(range(4, 50), erreur_moyenne)
# show()

# print "\nMonks-{}".format(n)
# erreur_moyenne = []
# for i in range(4, 50):
#     nn_monks = NeuralNet.NeuralNet(3, i, 6, 1)
#     for j in range(int(k)):
#         nn_monks.train(train_monks, train_labels_monks)
#     erreur_moyenne.append(nn_monks.test(test_monks, test_labels_monks))

# plot(range(4, 50), erreur_moyenne)
# show()


#Nombre de couches
# for i in range(3, 8):
#     print "i:  ", i
#     nn_iris = NeuralNet.NeuralNet(i, 4, 4, 1)
#     for j in range(int(k)):
#         nn_iris.train(train_iris, train_labels_iris)
#     nn_iris.test(test_iris, test_labels_iris)

# print "\nHouse votes"
# for i in range(3, 8):
#     print "i:  ", i
#     nn_votes = NeuralNet.NeuralNet(i, 11, 16, 1)
#     for j in range(int(k)):
#         nn_votes.train(train_votes, train_labels_votes)
#     nn_votes.test(test_votes, test_labels_votes)

# print "\nMonks-{}".format(n)
# for i in range(3, 8):
#     print "i:  ", i
#     nn_monks = NeuralNet.NeuralNet(i, 15, 6, 1)
#     for j in range(int(k)):
#         nn_monks.train(train_monks, train_labels_monks)
#     nn_monks.test(test_monks, test_labels_monks)

