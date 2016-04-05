import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

def clean_review(review,stopwords):
    result = ""
    lemmatizer = WordNetLemmatizer()
    for word in review:
        word = lemmatizer.lemmatize(word)
        if word not in stopwords:
            word = re.sub('[^A-Za-z ]','',word)
            if(len(word) != 0):
                result += word+" "
    return result

def clean_review_set_file(inputfile, outputfile, stopwordfile):
    try:
        finput  = open(inputfile,"r")
    except IOError:
        print("Training set file not found...\n\n")
        raise IOError

    foutput = open(outputfile,"w")

    stop_words_list =  read_words(stopwordfile)

    lemmatizer = WordNetLemmatizer()

    for line in finput:
        line = line.decode('utf-8').lower()
        input_words = word_tokenize(line)
        foutput.write(input_words[0] + " ")
        result = clean_review(input_words[1:],stop_words_list)
        foutput.write(result)
        foutput.write('\n')

    finput.close()
    foutput.close()

def main():
    inputfile     = "../data/shuffled-data.txt"
    outputfile    = "stopwords-removed-data-nltk.txt"
    stopwordfile = "stopwords.txt"
    clean_review_set_file(inputfile, outputfile, stopwordfile)

if __name__ == "__main__":
    main()
