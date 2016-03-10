import re

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in line.split()]

inputfile = "shuffled-data.txt"
outputfile = "stopwords-removed-data.txt"
stopwordfile = "stopwords.txt"

fina = open(inputfile)
fout = open(outputfile,"w")

delete_list = read_words("stopwords.txt")

for line in fina:
    line = line.lower()
    for word in delete_list:
        line = re.sub('[^A-za-z]'+ format(word)+'[^A-za-z0-9]',' ',line);
        review_class = line[0:2]
        line = re.sub('[^A-Za-z ]','',line)
        line = review_class[0] + line + '\n'
    fout.write(line)
fina.close()
fout.close()
