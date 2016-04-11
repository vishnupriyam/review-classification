from model.train import train_multinomial_naive_bayes, train_multinomial_bigram_naive_bayes
from nltk.tokenize import word_tokenize
import math
from cleaneddata.remove_stopwords_nltk import clean_review_set_file
from vocabulary.createVocabulary import createvocabulary
from nltk import bigrams

#divide the given reviewset to
def FoldTen(reviews):
    TenFoldReviews = []
    length = len(reviews)
    for i in range(0,9):
        TenFoldReviews.append(reviews[i*length/10:(i+1)*length/10])
    TenFoldReviews.append(reviews[9*length/10:])
    return TenFoldReviews

#unfold all reviews expect the ith set to a review array
def Unfold(reviewSet,index):
    reviews = []
    for i in range(0,10):
        if (i != index):
            reviews.extend(reviewSet[i]);
    return reviews

#get the actual class of all reviews in the given review set
def actual_class(reviews):
    actual = []
    for review in reviews:
        actual.append(review[0]);
    return actual

#create a test set, ie, remove the actual classes
def create_test_set(reviews):
    test_reviews = []
    for review in reviews:
        test_reviews.append(review[1:])
    return test_reviews

#predict the class of set of reviews
def predict(testSet,PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob):
    predicted_class = []
    for review in testSet:
        negative_probab = math.log10(PN)
        positive_probab = math.log10(PP)
        review_words = word_tokenize(review)
        for w in review_words:
            if w in negative_probabilities:
                negative_probab = negative_probab + math.log10(negative_probabilities[w])
            else:
                negative_probab = negative_probab + math.log10(unseen_neg_prob)
            if w in positive_probabilities:
                positive_probab = positive_probab + math.log10(positive_probabilities[w])
            else:
                positive_probab = positive_probab + math.log10(unseen_pos_prob)
        if(negative_probab > positive_probab):
            result = '-'
        else:
            result = '+'
        predicted_class.append(result)
    return predicted_class

def bigram_predict(testSet,PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob):
    predicted_class = []
    for review in testSet:
        negative_probab = math.log10(PN)
        positive_probab = math.log10(PP)
        review_words = []
        review_words.append('*')
        review_words.extend(word_tokenize(review))
        review_words.append('$')
        review_bigrams = bigrams(review_words)
        for w in review_bigrams:
            bigram = w
            w = w[0]+" " +w[1]
            if w in negative_probabilities and w in positive_probabilities:
                negative_probab = negative_probab + math.log10(negative_probabilities[w])
                positive_probab = positive_probab + math.log10(positive_probabilities[w])
            else:
                if bigram[0] in negative_probabilities and bigram[0] in positive_probabilities:
                    if(bigram[0] == '*'):
                        negative_probab = negative_probab
                        positive_probab = positive_probab
                    else:
                        #if(negative_probabilities[bigram[0]] < 0 or positive_probabilities[bigram[0]] < 0):
                        #    print("issue with " + bigram[0] + " " + str(negative_probabilities[bigram[0]]) + " " + str(positive_probabilities[bigram[0]]))
                        #if(negative_probabilities[bigram[0]] > 0 and positive_probabilities[bigram[0]] > 0):
                        negative_probab = negative_probab + math.log10(negative_probabilities[bigram[0]])
                        positive_probab = positive_probab + math.log10(positive_probabilities[bigram[0]])
                else:
                    negative_probab = negative_probab + math.log10(unseen_neg_prob)
                    positive_probab = positive_probab + math.log10(unseen_pos_prob)

        if(negative_probab > positive_probab):
            result = '-'
        else:
            result = '+'
        predicted_class.append(result)
    return predicted_class


def accuracy(actual_classification,predicted_classification):
    length = len(actual_classification)
    lengtht = len(predicted_classification)
    if(length != lengtht):
        print('length not same')
    count = 0;
    for i in range(0,length):
        if (actual_classification[i] == predicted_classification[i]):
            count = count + 1;
    acc = float(count)/float(length)
    return acc;

def validate(reviewfile, vocabfile, stopwordfile,bigram = False):
    reviews = []
    clean_review_set_file(reviewfile, "cleaneddata/stopwords-removed-data-nltk.txt", stopwordfile)
    freview = open("cleaneddata/stopwords-removed-data-nltk.txt","r")
    for line in freview:
        reviews.append(line.rstrip(' \n'));

    reviewSet = FoldTen(reviews)

    vocabulary = []
    try:
        fvocab = open(vocabfile,"r")
    except IOError:
        print("vocabulary for the given training set not found...\nGenerating Vocabulary...\n")
        createvocabulary("cleaneddata/stopwords-removed-data-nltk.txt", vocabfile)
        fvocab  = open(vocabfile,"r")

    for line in fvocab:
        vocabulary.append(line.rstrip('\n'))

    if(bigram):
        readmefile = "BigramOutput.txt"
    else:
        readmefile = "UnigramOutput.txt"
    freadme = open(readmefile,"w")

    avg_acc = 0

    for i in range(0,10):
        actual_classification = actual_class(reviewSet[i])
        testSet = create_test_set(reviewSet[i])
        trainingSet = Unfold(reviewSet,i)
        if(bigram == True):
            (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = train_multinomial_bigram_naive_bayes(trainingSet,vocabulary)
            predicted_classification = bigram_predict(testSet,PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob)
        else:
            (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = train_multinomial_naive_bayes(trainingSet,vocabulary)
            predicted_classification = predict(testSet,PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob)
        acc_result = accuracy(actual_classification,predicted_classification)
        freadme.write('run ' + str(i+1) + ": ")
        freadme.write(str(acc_result) + "\n\n")
        avg_acc += acc_result

    avg_acc /= 10

    freadme.write('average accuracy ' + str(avg_acc) + "\n\n")
    freadme.close()
    fvocab.close()
    freview.close()
    print("\n\nThe results have been successfully evaluated and been written into README.md file\n")

def main():
    reviewfile = "../cleaneddata/stopwords-removed-data-nltk.txt"
    vocabfile = "../vocabulary/vocabulary2.txt"
    stopwordfile = "../cleaneddata/stopwords.txt"
    validate(reviewfile, vocabfile, stopwordfile)

if __name__ == "__main__":
    main()
