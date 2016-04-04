from train import train_multinomial_naive_bayes
from nltk.tokenize import word_tokenize
import math

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

reviewfile = "../cleaned-data/stopwords-removed-data-nltk.txt"

reviews = []
freview = open(reviewfile,"r")
for line in freview:
    reviews.append(line.rstrip(' \n'));

reviewSet = FoldTen(reviews)

vocabulary = []
vocabfile = "../vocabulary/vocabulary2.txt"
fvocab = open(vocabfile,"r")
for line in fvocab:
    vocabulary.append(line.rstrip('\n'))

readmefile = "README.md"
freadme = open(readmefile,"w")

for i in range(0,10):
    actual_classification = actual_class(reviewSet[i])
    testSet = create_test_set(reviewSet[i])
    trainingSet = Unfold(reviewSet,i)
    (PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob) = train_multinomial_naive_bayes(trainingSet,vocabulary)
    predicted_classification = predict(testSet,PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob)
    acc_result = accuracy(actual_classification,predicted_classification)
    freadme.write('run ' + str(i+1) + ": ")
    freadme.write(str(acc_result) + "\n")
