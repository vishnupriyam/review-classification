from collections import Counter
from nltk import word_tokenize, bigrams

'''
    Creates a vocabulary and write the words in vocabulary to the vocabfile(unigrams with count >= 2)
    @param - reviewfile - path to file containing all reviews
    @param - vocabfile  - path to vocabulary file
'''

def createvocabulary(reviewfile,vocabfile):

    finput = open(reviewfile,"r")
    foutput = open(vocabfile,"w")

    all_words      = []
    for line in finput:
        words = line.split();
        all_words.extend(words)

    c = Counter(all_words)

    for w in c:
       if (w != "+" and w != "-" and c[w] >=2):
           foutput.write(w);
           foutput.write('\n');

    finput.close()
    foutput.close()

'''
    Creates a vocabulary and write the words in vocabulary to the vocabfile (unigrams with count >= 2, bigrams with count>=3)
    @param - reviewfile - path to file containing all reviews
    @param - vocabfile  - path to vocabulary file
'''

def createbigramvocabulary(reviewfile, vocabfile):
    createvocabulary(reviewfile, vocabfile)
    finput = open(reviewfile,"r")
    foutput = open(vocabfile,"a")

    all_bigrams = []
    for line in finput:
        tokenized_line = []
        tokenized_line.append('*')
        tokenized_line.extend(word_tokenize(line[1:]))
        tokenized_line.append('$')
        bgrms = bigrams(tokenized_line)
        all_bigrams.extend(bgrms)

    c = Counter(all_bigrams)

    for b in c:
        if (b[0] != "+" and b[0] != "-" and c[b] >= 3):
            foutput.write(b[0] + " " + b[1] + "\n")

    finput.close()
    foutput.close()

def main():
    reviewfile = "../cleaneddata/stopwords-removed-data-nltk.txt"
    vocabfile  = "vocabulary3.txt"
    createbigramvocabulary(reviewfile,vocabfile)

if __name__ == "__main__":
    main()
