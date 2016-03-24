import re
from nltk.tokenize import sent_tokenize, word_tokenize

inputfile     = "../data/shuffled-data.txt"
outputfile    = "stopwords-removed-data-nltk.txt"

finput  = open(inputfile,"r");
foutput = open(outputfile,"w");

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

stop_words_list =  read_words("stopwords.txt");

line = "it's i'm don't"
words = word_tokenize(line)

for line in finput:
    line = line.decode('utf-8').lower();
    input_words = word_tokenize(line);
    foutput.write(input_words[0] + " ");
    input_words = input_words[1:];
    for word in input_words:
        if word not in stop_words_list:
            word = re.sub('[^A-Za-z ]','',word);
            if(len(word) != 0):
                foutput.write(word+" ");
    foutput.write('\n');

finput.close();
foutput.close();
