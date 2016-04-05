from validation.validate import validate

reviewfile = raw_input("Enter the review training set file path : ")
vocabfile = raw_input("Enter the vocabulary file path : ")
stopwordfile = raw_input("Enter the stopword file path : ")
validate(reviewfile, vocabfile, stopwordfile)
