
#TODO improve the program so that it implements all the tasks of removing stopwords creating vocab when the miput file is just the raw data

import sys
import imp
import pickle

from train import train_multinomial_naive_bayes
from vocabulary.createVocabulary import createvocabulary

def generatemodel(reviewfile,vocabfile,paramfile):
    try:
        freview = open(reviewfile,"r")
    except IOError:
        print("Training set file not found...\n\n")
        raise IOError

    try:
        fvocab  = open(vocabfile,"r")
    except IOError:
        print("vocabulary for the given training set not found...\nGenerating Vocabulary...\n")
        createvocabulary(reviewfile, vocabfile)
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
    reviewfile = "../cleaneddata/stopwords-removed-data-nltk.txt"
    vocabfile = "../vocabulary/vocabulary2.txt"
    paramfile = "parameters.p"
    generatemodel(reviewfile, vocabfile,paramfile)

if __name__ == "__main__":
    main()
