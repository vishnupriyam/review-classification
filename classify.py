import sys
import pickle
from cleaned-data import *
from model.generateModel import generatemodel
import validation.validate

review = raw_input("Enter a review to classify: ")
print review
try:
    paramfile = open("model/parameters.p", "r")
except IOError:
    print("No saved model found...\nGenerating a new model...\n")
    reviewfile = raw_input("Enter the review training set file path : ")
    vocabfile = raw_input("Enter the vocabulary file path : ")
    generatemodel(reviewfile, vocabfile, "model/parameters.p")

(PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = pickle.load( open("model/parameters.p","rb") )

#clean the input review
