from collections import Counter

finput = open("../cleaned-data/stopwords-removed-data-nltk.txt","r");
foutput = open("vocabulary2.txt","w");

positive_words = []
negative_words = []
all_words      = []
for line in finput:
    words = line.split();
    if(words[0] == "+"):
        positive_words.extend(words[1:])
    else:
        negative_words.extend(words[1:])
    all_words.extend(words)

lem_words = []

for w in all_words:
    lem_words.append(w)
pos_lem_words = []
for w in positive_words:
    pos_lem_words.append(w)
neg_lem_words = []
for w in negative_words:
    neg_lem_words.append(w)

c = Counter(lem_words)
cp = Counter(pos_lem_words)
cn = Counter(neg_lem_words)

for w in c:
   if (w != "+" and w != "-" and c[w] >=2):
       foutput.write(w);
       #foutput.write(" ");
       #foutput.write('{}'.format(cp[w]) + " ");
       #foutput.write('{}'.format(cn[w]) + " ");
       #foutput.write('{}'.format(c[w]) + " ");
       foutput.write('\n');
