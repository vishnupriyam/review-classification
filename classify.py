import sys
import validation.validate
from model.generateModel import generatemodel

review = raw_input("Enter a review to classify: ")
try:
    paramfile = open("parameters.p", "r")
except IOError:
    print("No saved model found...\nGenerating a new model...\n")
    reviewfile = raw_input("Enter the review training set file path : ")
    vocabfile = raw_input("Enter the vocabulary file path : ")
    generatemodel(reviewfile, vocabfile, "parameters.p")
