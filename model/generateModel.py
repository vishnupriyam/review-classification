
#TODO improve the program so that it implements all the tasks of removing stopwords creating vocab when the miput file is just the raw data

import sys
import imp
import pickle

from train import train_multinomial_naive_bayes

def generatemodel(reviewfile,vocabfile,paramfile):

    freview = open(reviewfile,"r")
    fvocab  = open(vocabfile,"r")

    reviews = []
    vocabulary = []
    for line in freview:
        reviews.append(line.rstrip('\n'))
    for line in fvocab:
        vocabulary.append(line.rstrip('\n'))

    (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = train_multinomial_naive_bayes(reviews,vocabulary)

    training_model = (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob)

    pickle.dump(training_model,open(paramfile,"w"))

def main():
    reviewfile = "../cleaned-data/stopwords-removed-data-nltk.txt"
    vocabfile = "../vocabulary/vocabulary2.txt"
    paramfile = "parameters.p"
    generatemodel(reviewfile, vocabfile,paramfile)

if __name__ == "__main__":
    main()
