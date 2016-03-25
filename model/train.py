from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def train_multinomial_naive_bayes(reviews, vocabulary):

    positive_reviews = []
    negative_reviews = []

    lemmatizer = WordNetLemmatizer()
    lem_words = []

    no_of_positive_reviews = 0
    no_of_negative_reviews = 0


    for line in reviews:
        words = word_tokenize(line[1:])
        for w in words:
            w = lemmatizer.lemmatize(w)
            if(line[0] == "+"):
                positive_reviews.append(w)
            else:
                negative_reviews.append(w)
        if(line[0]=="+"):
            no_of_positive_reviews+=1
        else:
            no_of_negative_reviews+=1

    #probability of a class c P(c) = no_of_class_c_reviews / total_no_of_reviews

    #probability of class positive
    PP = float(no_of_positive_reviews) / float(no_of_positive_reviews + no_of_negative_reviews)
    #porbability of class negative
    PN = float(no_of_negative_reviews) / float(no_of_positive_reviews + no_of_negative_reviews)

    vocab_size = len(vocabulary)

    #count(c) = total count of all words in vocabulary in class c file
    count_p = 0;
    count_n = 0;

    #do word tokenisze beeterr else substrings also counted

    for word in vocabulary:
        count_p = count_p + positive_reviews.count(word)
        count_n = count_n + negative_reviews.count(word)

    #P(W|c) = (count(W,c)+1)/(count(c)+|V|)

    positive_probabilities = {}
    negative_probabilities = {}

    for word in vocabulary:
        PPW = positive_reviews.count(word)
        PNW = negative_reviews.count(word)

        positive_probabilities[word] = float(PPW + 1)/float(count_p + vocab_size)
        negative_probabilities[word] = float(PNW + 1)/float(count_n + vocab_size)


    print(PP)
    print(PN)
    print(positive_probabilities)
    return(PP,PN,positive_probabilities,negative_probabilities);
