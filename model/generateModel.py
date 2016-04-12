import sys
import imp
import pickle

from train import train_multinomial_naive_bayes, train_multinomial_bigram_naive_bayes
from vocabulary.createVocabulary import createvocabulary
from cleaneddata.remove_stopwords_nltk import clean_review_set_file

'''
    Generates a Multinomial Naive Bayes model with training data in reviewfile, vocabulary in vocabfile and saves the model using pickle dump to paramfile.
    @param : reviewfile - path to the file containing training data
    @param : vocabfile  - path to the file containing vocabulary for the model
    @param : paramfile  - path to the file where model is to be dumped
    @param : bigram     - if True,  function uses multinomial naive bayes with bigram model
                          if False, function uses multinomial naive bayes with unigram model
'''
def generatemodel(reviewfile,vocabfile,paramfile,bigram = False):
    #clean the training data
    clean_review_set_file(reviewfile, "cleaneddata/stopwords-removed-data-nltk.txt", "cleaneddata/stopwords.txt")
    freview = open("cleaneddata/stopwords-removed-data-nltk.txt", "r")

    try:
        fvocab  = open(vocabfile,"r")
    except IOError:
        print("vocabulary for the given training set not found...\nGenerating Vocabulary...\n")
        createvocabulary("cleaneddata/stopwords-removed-data-nltk.txt", vocabfile)
        fvocab  = open(vocabfile,"r")

    #read reviews from file into an array
    reviews = []
    vocabulary = []
    for line in freview:
        reviews.append(line.rstrip('\n'))
    #read words from vocabulary
    for line in fvocab:
        vocabulary.append(line.rstrip('\n'))

    #choose the training type according to the value for bigram
    if(bigram):
        #bigram model
        (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = train_multinomial_bigram_naive_bayes(reviews,vocabulary)
    else:
        #unigram model
        (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = train_multinomial_naive_bayes(reviews,vocabulary)

    training_model = (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob)

    #dump parameters into paramfile
    pickle.dump(training_model,open(paramfile,"w"))

def main():
    reviewfile = "../cleaneddata/stopwords-removed-data-nltk.txt"
    vocabfile = "../vocabulary/vocabulary2.txt"
    paramfile = "savedmodel.p"
    generatemodel(reviewfile, vocabfile,paramfile,False)

if __name__ == "__main__":
    main()
