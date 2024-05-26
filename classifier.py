import math
import nltk

from nltk.corpus import stopwords
import spacy
from spacy import load
from spacy.lang.ru import Russian

nlp = Russian()
load_model = load("ru_core_news_sm")


def preprocessing_lemm(text):
    sentences = nltk.sent_tokenize(text)
    words = []
    for doc in load_model.pipe(sentences):
        words += [n.lemma_ for n in doc]
    stop_words = set(stopwords.words('russian'))
    words = [word for word in words if word not in stop_words]
    words = [word.lower() for word in words if word.isalpha()]
    return words


def preprocessing_stem(text):
    tokenized_text = nltk.word_tokenize(text)
    words = []
    stemmer = nltk.stem.snowball.RussianStemmer()
    for word in tokenized_text:
        words.append(stemmer.stem(word))
    stop_words = set(stopwords.words('russian'))
    words = [word for word in words if word not in stop_words]
    words = [word.lower() for word in words if word.isalpha()]
    return words


def calcw1(N1, N2, N3, m, s, wSize):
    if N1 == wSize:
        return 1
    return 1 + (m - 1) * ((N3 + N2) / (wSize - N1)) ** (1 / s)


text = open('data/борьба.txt', encoding='utf-8', mode='r').read()

words = preprocessing_lemm(text)

d1t = open('dictionaries/d1.txt', encoding='utf-8', mode='r').read()
d2t = open('dictionaries/d2.txt', encoding='utf-8', mode='r').read()
d3t = open('dictionaries/d3.txt', encoding='utf-8', mode='r').read()
D1 = preprocessing_lemm(d1t)
D2 = preprocessing_lemm(d2t)
D3 = preprocessing_lemm(d3t)

wSize = 40
displacement = 10
w2 = 20
w3 = 30
s = 3
m = 15
f = []
threshold_value = 0.12

for i in range((len(words) - 40) // displacement):
    window = words[(i * displacement):(i * displacement) + wSize]
    weights = dict([(word, 0) for word in window])
    wordsCount = dict([(word, 0) for word in window])
    N1 = N2 = N3 = 0
    for j, word in enumerate(window):
        wordsCount[word] += 1
        if word in D1:
            weights[word] = -1
            N1 += 1
        elif word in D2:
            weights[word] = w2
            N2 += 1
        elif word in D3:
            weights[word] = w3
            N3 += 1
    w1 = calcw1(N1, N2, N3, m, s, wSize)
    for word in weights.keys():
        if weights[word] == -1:
            weights[word] = w1
    for word in wordsCount.keys():
        if wordsCount[word] > 1:
            weights[word] *= math.sqrt(wordsCount[word])
    f.append(sum(weights.values()) / (w3 * wSize))

print(max(f))
if max(f) > threshold_value:
    print('текст экстремистский')
