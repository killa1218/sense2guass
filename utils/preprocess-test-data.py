# coding=utf8

from nltk.stem import WordNetLemmatizer
import re
import pickle as pk

'''
[
    {
        w1: String,
        w2: String,
        c1: [String],
        c2: [String],
        r: float
    }, ...
]
'''

lemmatizer = WordNetLemmatizer()
p = re.compile(' (\W|<b>|</b>)+ ')
res = []

with open('../data/SCWS/ratings.txt') as f:
    for line in f.readlines():
        d = {}
        arr = line.split('\t')
        d['w1'] = lemmatizer.lemmatize(arr[1].lower(), pos=arr[2])
        d['w2'] = lemmatizer.lemmatize(arr[3].lower(), pos=arr[4])

        c1 = re.sub(' (\W|<b>|</b>)+ ', ' ', arr[5])
        c1 = re.sub(' \(|\) |\( |\) ', ' ', c1)
        c1List = []


        for w in c1.split(' '):
            if w == d['w1']:
                c1List.append(lemmatizer.lemmatize(w, pos=arr[2]).lower())
            else:
                c1List.append(lemmatizer.lemmatize(w).lower())

        c2 = re.sub(' (\W|<b>|</b>)+ ', ' ', arr[6])
        c2 = re.sub(' \(|\) |\( |\) ', ' ', c2)
        c2List = []

        for w in c2.split(' '):
            if w == d['w2']:
                c2List.append(lemmatizer.lemmatize(w, pos=arr[4]).lower())
            else:
                c2List.append(lemmatizer.lemmatize(w).lower())

        d['c1'] = c1List
        d['c2'] = c2List
        d['r'] = float(arr[7])

        res.append(d)

print(res)
f = open('../data/SCWS/testData.pk3', 'wb')
pk.dump(res, f)
