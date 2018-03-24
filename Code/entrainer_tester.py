import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn

# init param




# inti classifier

print "\n================= I N I T =================\n"

k = 5
knn_iris = Knn.Knn(k)
knn_votes = Knn.Knn(k)
knn_monks = Knn.Knn(k)

bayesNaif_iris = BayesNaif.BayesNaif()
bayesNaif_votes = BayesNaif.BayesNaif()
bayesNaif_monks = BayesNaif.BayesNaif()

# load datasets

print "\n================= L O A D =================\n"

n= 1;

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(0.5)
# train_votes, train_labels_votes, test_votes, test_labels_votes = load_datasets.load_congressional_dataset(0.5)
# train_monks, train_labels_monks, test_monks, test_labels_monks = load_datasets.load_monks_dataset(n)

# train

print "\n================= T R A I N =================\n"

print "\nKNN\n"

print "Bezdek Iris"
knn_iris.train(train_iris, train_labels_iris)

# print "\nHouse votes"
# knn_votes.train(train_votes, train_labels_votes)

# print "\nMonks-"
# knn_monks.train(train_monks, train_labels_monks)

print "\nBAYES NAIF\n"

print "Bezdek Iris"
bayesNaif_iris.train(train_iris, train_labels_iris)

# print "\nHouse votes"
# bayesNaif_votes.train(train_votes, train_labels_votes)

# print "\nMonks-"
# bayesNaif_monks.train(train_monks, train_labels_monks)


# test

print "\n================= T E S T =================\n"

print "\nKNN\n"

print "Bezdek Iris"
knn_iris.test(test_iris, test_labels_iris)

# print "\nHouse votes"
# knn_votes.test(test_votes, test_labels_votes)

# print "\nMonks-" + `n`
# knn_monks.test(test_monks, test_labels_monks)

print "\nBAYES NAIF\n"

print "Bezdek Iris"
bayesNaif_iris.test(test_iris, test_labels_iris)

# print "\nHouse votes"
# bayesNaif_votes.test(test_votes, test_labels_votes)

# print "\nMonks-" + `n`
# bayesNaif_monks.test(test_monks, test_labels_monks)




