# coding=utf8

from nltk.stem import WordNetLemmatizer
import re
import pickle as pk


le = WordNetLemmatizer()

def processSCWS():
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
    res = []

    with open('../data/SCWS/ratings.txt') as f:
        for line in f.readlines():
            d = {}
            arr = line.split('\t')
            d['w1'] = le.lemmatize(arr[1].lower(), pos=arr[2])
            d['w2'] = le.lemmatize(arr[3].lower(), pos=arr[4])

            c1 = re.sub(' (\W|<b>|</b>)+ ', ' ', arr[5])
            c1 = re.sub(' \(|\) |\( |\) ', ' ', c1)
            c1List = []


            for w in c1.split(' '):
                if w.lower() == arr[1].lower():
                    c1List.append(le.lemmatize(w.lower(), pos=arr[2]))
                else:
                    c1List.append(le.lemmatize(w.lower()))

            c2 = re.sub(' (\W|<b>|</b>)+ ', ' ', arr[6])
            c2 = re.sub(' \(|\) |\( |\) ', ' ', c2)
            c2List = []

            for w in c2.split(' '):
                if w.lower() == arr[3].lower():
                    c2List.append(le.lemmatize(w.lower(), pos=arr[4]))
                else:
                    c2List.append(le.lemmatize(w.lower()))

            d['c1'] = c1List
            d['c2'] = c2List
            d['r'] = float(arr[7])

            res.append(d)

    print(res)
    f = open('../data/SCWS/testData.pk3', 'wb')
    pk.dump(res, f)
    f.close()


def processBLESS():
    '''
    [
        {
            w: String,
            mero: [String]
            hyper: [String]
            r: [String]
            c: String
        }, ...
    ]
    '''
    res = []

    with open('../data/BLESS/BLESS.txt') as f:
        prevWord = None
        mero = []
        hyper = []
        r = []
        cate = ''
        tmp = {}
        for line in f.readlines():
            lineArr = line.strip().split('\t')
            word = lineArr[0].lower()
            type = lineArr[2]
            word2 = lineArr[3].lower()

            if len(word2.split('-')) > 2:
                continue

            try:
                word = le.lemmatize(word.split('-')[0], 'a' if word.split('-')[1] == 'j' else word.split('-')[1])
                word2 = le.lemmatize(word2.split('-')[0], 'a' if word2.split('-')[1] == 'j' else word2.split('-')[1])
            except:
                print(word, word2)
                return

            if prevWord != word:
                if prevWord:
                    tmp['w'] = prevWord
                    tmp['mero'] = mero
                    tmp['hyper'] = hyper
                    tmp['r'] = r
                    tmp['c'] = cate

                    res.append(tmp)

                prevWord = word
                cate = lineArr[1]
                mero = []
                hyper = []
                r = []
                tmp = {}

            if type == 'mero':
                mero.append(word2)
            elif type == 'hyper':
                hyper.append(word2)
            elif 'random' in type:
                r.append(word2)

        tmp['w'] = prevWord
        tmp['mero'] = mero
        tmp['hyper'] = hyper
        tmp['r'] = r
        tmp['c'] = cate

    with open('../data/BLESS/bless.pk3', 'wb') as f:
        print(res)
        pk.dump(res, f)

if __name__ == '__main__':
    # processBLESS()
    processSCWS()
