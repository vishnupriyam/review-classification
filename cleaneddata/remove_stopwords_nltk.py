import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

'''
    Returns array of words in the given file
    @param:  words_file - path to file
    @return: array of words in the given file
'''
def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

'''
    Returns review after the cleaning process
    @param: review - a review
    @param: stopwords - array of stopwords to be removed from review
    @return: result - cleaned review
'''
def clean_review(review,stopwords):
    result = ""
    lemmatizer = WordNetLemmatizer()
    for word in review:
        #converts the word to its lemma form
        word = lemmatizer.lemmatize(word)
        #adds the word to the resultant review only if its not a stopword
        if word not in stopwords:
            #removes all non-alphabet characters
            word = re.sub('[^A-Za-z ]','',word)
            if(len(word) != 0):
                result += word+" "
    return result

'''
    Cleans the reviews in the inputfile and writes the result to the outputfile
    @param : inputfile - path to inputfile that is to be cleaned
    @param : outputfile - path to outputfile for clean data
    @param : stopwordfile - path to file with list of stopwords
'''
def clean_review_set_file(inputfile, outputfile, stopwordfile):
    try:
        finput  = open(inputfile,"r")
    except IOError:
        print("Training set file not found...\n\n")
        raise IOError

    foutput = open(outputfile,"w")

    stop_words_list =  read_words(stopwordfile)

    seen = set()
    for line in finput:
        #decode the review to utf-8 form and converts the it to lowercase
        line = line.decode('utf-8').lower()
        #tokenize the review
        input_words = word_tokenize(line)
        #clean the review
        result = clean_review(input_words[1:],stop_words_list)
        #adds to the outputfile only if review is not a duplicate that was formed due to crawling process
        if result not in seen:
            foutput.write(input_words[0] + " ")
            foutput.write(result)
            seen.add(result)
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
