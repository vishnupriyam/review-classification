import sys
import pickle
from nltk.tokenize import word_tokenize
from cleaneddata.remove_stopwords_nltk import read_words, clean_review
from model.generateModel import generatemodel
from validation.validate import predict

review = raw_input("Enter a review to classify: ")

try:
    paramfile = open("model/parameters.p", "r")
except IOError:
    print("No saved model found...\nGenerating a new model...\n")
    reviewfile = raw_input("Enter the review training set file path : ")
    vocabfile = raw_input("Enter the vocabulary file path : ")
    generatemodel(reviewfile, vocabfile, "model/parameters.p")

#model
(PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = pickle.load( open("model/parameters.p","rb") )

#clean the input review
review = review.lower()
review = word_tokenize(review)
stopwords = read_words("cleaneddata/stopwords.txt")
review = clean_review(review,stopwords)

#predict the class of the review
testreview = []
testreview.append(review)
predicted_class = predict(testreview,PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob)
print(predicted_class)
