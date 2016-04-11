from validation.validate import validate

#reviewfile = raw_input("Enter the review training set file path : ")
#vocabfile = raw_input("Enter the vocabulary file path : ")
#stopwordfile = raw_input("Enter the stopword file path : ")
reviewfile = "data/shuffled-data.txt"
vocabfile = "vocabulary/vocabulary3.txt"
stopwordfile = "cleaneddata/stopwords.txt"
validate(reviewfile, vocabfile, stopwordfile,True)
