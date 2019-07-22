import nltk
lemmatizer = nltk.stem.WordNetLemmatizer()


f = open("sentences.txt", "r")
for x in f:
    print(x)
    tokenized = nltk.tokenize.word_tokenize(x.lstrip().rstrip())
    print(tokenized)
    pos_tagged = [p[1] for p in nltk.pos_tag(tokenized)]
    print(pos_tagged)
    lemmatized = [lemmatizer.lemmatize(tokenized[i])
               if not pos_tagged[i].startswith("V") else lemmatizer.lemmatize(tokenized[i], pos='v')
              for i in range(len(tokenized))]
    print(lemmatized)
