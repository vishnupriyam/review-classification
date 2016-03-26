vocabfile = "vocabulary.txt"

fvocab = open(vocabfile,"r")

count = []
total = 0;

for line in fvocab:
    line = line.split(" ")
    count.append(line[1]);
    total+=int(line[1])

#print(count);
print(total)

#obtained positive total = 27083
