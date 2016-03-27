
#TODO improve the program so that it implements all the tasks of removing stopwords creating vocab when the miput file is just the raw data

import sys
import imp
import pickle

from train import train_multinomial_naive_bayes

check = len(sys.argv)
if(check > 1):
    reviewfile = sys.argv[1]
    vocabfile = sys.arg[2]
else:
    reviewfile = "../cleaned-data/stopwords-removed-data-nltk.txt"
    vocabfile  = "../vocabulary/vocabulary2.txt"

freview = open(reviewfile,"r")
fvocab  = open(vocabfile,"r")

reviews = []
vocabulary = []
for line in freview:
    reviews.append(line.rstrip('\n'))
for line in fvocab:
    vocabulary.append(line.rstrip('\n'))

(PP,PN,positive_probabilities,negative_probabilities) = train_multinomial_naive_bayes(reviews,vocabulary)

training_model = (PP,PN,positive_probabilities,negative_probabilities)

pickle.dump(training_model,open("parameters.p","w"))
