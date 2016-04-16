Code:

Stage 1 : file with labelled reviews - data/shuffled-data.txt
Stage 2 : code - cleaneddata/remove_stopwords_nltk.py
          data file after cleaning - stopwords-removed-data-nltk.txt
Stage 3 : code - vocabulary/createVocabulary.py and the function is createvocabulary
          vocabulary file - vocabulary/vocabulry2.txt
Stage 4 : code - model/train.py and function is train_multinomial_naive_bayes and model is generated using model/generateModel.py
Stage 5 : code - validate.py which uses validation/validate.py
          Unigram accuracy results - UnigramOutput.txt
Stage 6 : code - vocabulary/createVocabulary.py and the function is createbigramvocabulary
          vocabulary file - vocabulary/vocabulary3.txt
Stage 7 : code - model/train.py and function is train_multinomial_bigram_naive_bayes
          validation - validate.py uses validation/validate.py and function used is bigram_predict

Unigram Accuracy
----------------

run 1: 0.785046728972<br/>
run 2: 0.824074074074<br/>
run 3: 0.861111111111<br/>
run 4: 0.878504672897<br/>
run 5: 0.898148148148<br/>
run 6: 0.898148148148<br/>
run 7: 0.88785046729<br/>
run 8: 0.861111111111<br/>
run 9: 0.796296296296<br/>
run 10: 0.888888888889<br/>

average accuracy 0.857917964694<br/>

Bigram Accuracy
---------------

run 1: 0.859813084112<br/>
run 2: 0.842592592593<br/>
run 3: 0.87962962963<br/>
run 4: 0.915887850467<br/>
run 5: 0.888888888889<br/>
run 6: 0.87962962963<br/>
run 7: 0.878504672897<br/>
run 8: 0.87037037037<br/>
run 9: 0.851851851852<br/>
run 10: 0.861111111111<br/>

average accuracy 0.872827968155<br/>


Probability calculations
------------------------
Probability P(c|d)= P(d|c)\*P(c)/P(d)
based on which probability P(c1|d) or P(c2|d), class c1(pos) or c2(neg) is chosen...
P(d|c) is calculated as:
    P(d|c)=P(w1,w2,....w(n)|c) =P(w1,w2|c)\*P(w2,w3|c)....\*P(w(n-1),w(n)|c)   (Assumption is all conditional probabilities are independent...)
      For Bigram model...
      P(w1,w2|c)=(count(w1,w2|c)+1)/(count(c)+|V|)
      For Unigram model...
        Probablity of P(w|c)=(count(w,c)+1)/(count(c)+|V|)
            count(w,c) = count of word w in all documents of class c
            count(c)   = count of words in class c
            |V|        = total distint words in the our trainig set
           used add one smoothing
