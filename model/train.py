from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import bigrams

def train_multinomial_naive_bayes(reviews, vocabulary):

    positive_reviews = []
    negative_reviews = []

    lem_words = []

    no_of_positive_reviews = 0
    no_of_negative_reviews = 0


    for line in reviews:
        words = word_tokenize(line[1:])
        for w in words:
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
    p_counter = Counter(positive_reviews)
    n_counter = Counter(negative_reviews)

    #do counter better if positive review is a single string then substrings are also counted for a given word

    for word in vocabulary:
        count_p = count_p + p_counter[word]
        count_n = count_n + n_counter[word]

    #P(W|c) = (count(W,c)+1)/(count(c)+|V|)

    positive_probabilities = {}
    negative_probabilities = {}

    for word in vocabulary:
        PPW = p_counter[word]
        PNW = n_counter[word]

        positive_probabilities[word] = float(PPW + 1)/float(count_p + vocab_size)
        negative_probabilities[word] = float(PNW + 1)/float(count_n + vocab_size)

    unseen_pos_prob = 1/float(count_p + vocab_size)
    unseen_neg_prob = 1/float(count_n + vocab_size)

    return(PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob);


'''
    Probability P(c|d)= P(d|c)*P(c)/P(d)
    based on which probability P(c1|d) or P(c2|d), class c1(pos) or c2(neg) is chosen...
    P(c) is the prior probability ( P(c)=0.5 ) for our training corpus (containing equal no. of +ve and -ve reviews)
    P(d|c) is calculated as:
        P(d|c)=P(w1,w2,....w(n)|c) =P(w1,w2|c)*P(w2,w3|c)....*P(w(n-1),w(n)|c)   (Assumption is all conditional probabilities are independent...)
          For Bigram model...
          P(w1,w2|c)=(count(w1,w2|c)+1)/(count(c)+|V|)
          For Unigram model...
            Probablity of P(w|c)=(count(w,c)+1)/(count(c)+|V|)
                count(w,c) = count of word w in all documents of class c
                count(c)   = count of words in class c
                |V|        = total distint words in the our trainig set
               used add one smoothing
'''

def train_multinomial_bigram_naive_bayes(reviews, vocabulary):
    positive_reviews = []
    positive_bigram_reviews = []
    negative_reviews = []
    negative_bigram_reviews = []

    no_of_positive_reviews = 0
    no_of_negative_reviews = 0

    for line in reviews:
        words = word_tokenize(line[1:])
        bigram_set = bigrams(words)
        for w in words:
            if(line[0] == "+"):
                positive_reviews.append(w)
                for bigram in bigram_set:
                    positive_bigram_reviews.append(bigram[0]+" "+bigram[1])
            else:
                negative_reviews.append(w)
                for bigram in bigram_set:
                    positive_bigram_reviews.append(bigram[0]+" "+bigram[1])
        if(line[0]=="+"):
            no_of_positive_reviews+=1
        else:
            no_of_negative_reviews+=1

    #probability of class positive
    PP = float(no_of_positive_reviews) / float(no_of_positive_reviews + no_of_negative_reviews)
    #porbability of class negative
    PN = float(no_of_negative_reviews) / float(no_of_positive_reviews + no_of_negative_reviews)

    #vocab_size = len(vocabulary)
    vocab_size = 0

    p_counter = Counter(positive_reviews)
    n_counter = Counter(negative_reviews)

    pb_counter = Counter(positive_bigram_reviews)
    nb_counter = Counter(negative_bigram_reviews)


    count_p = 0;
    count_n = 0;

    positive_probabilities = {}
    negative_probabilities = {}

    for word in vocabulary:
        temp = word.split()
        if(len(temp) == 1):
            count_p = count_p + p_counter[word]
            count_n = count_n + n_counter[word]
            vocab_size += 1

    for line in vocabulary:
        line_a = line.split()
        if(len(line_a) == 2):
            #p_counter[line_a[0]] -= pb_counter[line]
            #p_counter[line_a[1]] -= pb_counter[line]
            #n_counter[line_a[0]] -= nb_counter[line]
            #n_counter[line_a[1]] -= nb_counter[line]
            positive_probabilities[line] = float(pb_counter[line]+ 1)/float(count_p + vocab_size)
            negative_probabilities[line] = float(nb_counter[line]+ 1)/float(count_n + vocab_size)

    for line in vocabulary:
        line_a = line.split()
        if(len(line_a) == 1):
            PPW = p_counter[line]
            PNW = n_counter[line]

            positive_probabilities[line] = float(PPW + 1)/float(count_p + vocab_size)
            negative_probabilities[line] = float(PNW + 1)/float(count_n + vocab_size)

    unseen_pos_prob = 1/float(count_p + vocab_size)
    unseen_neg_prob = 1/float(count_n + vocab_size)

    return(PP,PN,positive_probabilities,negative_probabilities,unseen_pos_prob,unseen_neg_prob);
