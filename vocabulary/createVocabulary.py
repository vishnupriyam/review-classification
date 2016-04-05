from collections import Counter

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

def main():
    reviewfile = "../cleaned-data/stopwords-removed-data-nltk.txt"
    vocabfile  = "vocabulary2.txt"
    createvocabulary(reviewfile,vocabfile)

if __name__ == "__main__":
    main()
