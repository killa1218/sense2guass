# coding=utf8

from nltk import WordNetLemmatizer
import nltk
import re
import pickle as pk


le = WordNetLemmatizer()

def processText8():
    '''
        Do lemmatizing
    '''
    with open('../data/text8') as f, open('../data/text8lem', 'w') as wf:
        for line in f.readlines():
            arr = nltk.word_tokenize(line)
            print(len(arr))
            print(arr[:10000])
            tagged = nltk.pos_tag(arr)
            print(tagged[:10000])

            for w in tagged:
                if len(w[1]) > 0:
                    if w[1][0] == 'N':
                        wf.write(le.lemmatize(w[0], 'n'))
                    elif w[1][0] == 'J':
                        wf.write(le.lemmatize(w[0], 'a'))
                    elif w[1][0] == 'V' or w[1][0] == 'I':
                        wf.write(le.lemmatize(w[0], 'v'))
                    elif w[1][0] == 'R':
                        wf.write(le.lemmatize(w[0], 'r'))
                    else:
                        wf.write(le.lemmatize(w[0]))
                else:
                    wf.write(le.lemmatize(w[0]))

                wf.write(' ')


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
            meet = False
            wordFilter = re.compile('^[a-zA-Z]+(-[a-zA-Z]+)*$')
            puncFilter = re.compile('[,\.?:;\'\"!`]|(-{2})|(\.{3})|\(|\)|\[|\]|\{|\}')
            arr = line.split('\t')
            d['w1'] = le.lemmatize(arr[1].lower(), pos=arr[2])
            d['w2'] = le.lemmatize(arr[3].lower(), pos=arr[4])

            arr[5] = puncFilter.sub('', arr[5])
            arr[5] = arr[5].replace('<b>', '|||')
            arr[5] = arr[5].replace('</b>', '')
            c1 = nltk.pos_tag(nltk.word_tokenize(arr[5]))
            c1List = []

            for w in c1:
                if w[0] == '|||':
                    d['w1sIdx'] = len(c1List) # Find the index of target word.
                    meet = True
                else:
                    if wordFilter.match(w[0]):
                        if meet:
                            c1List.append(d['w1']) # Not lemmatize the main word in context
                            meet = False
                        elif len(w[1]) > 0:
                            if w[1][0] == 'N':
                                c1List.append(le.lemmatize(w[0], 'n').lower())
                            elif w[1][0] == 'J':
                                c1List.append(le.lemmatize(w[0], 'a').lower())
                            elif w[1][0] == 'V':
                                c1List.append(le.lemmatize(w[0], 'v').lower())
                            elif w[1][0] == 'R':
                                c1List.append(le.lemmatize(w[0], 'r').lower())
                            else:
                                c1List.append(le.lemmatize(w[0], 'v').lower())
                        else:
                            c1List.append(le.lemmatize(w[0], 'v').lower())
                    else:
                        pass
                        # print(w[0], 'Ignored')
            if c1List[d['w1sIdx']] != d['w1']:
                print(c1List[d['w1sIdx']], d['w1'])
            assert(c1List[d['w1sIdx']] == d['w1'])

            arr[6] = puncFilter.sub('', arr[6])
            arr[6] = arr[6].replace('<b>', '|||')
            arr[6] = arr[6].replace('</b>', '')
            c2 = nltk.pos_tag(nltk.word_tokenize(arr[6]))
            c2List = []

            for w in c2:
                if w[0] == '|||':
                    d['w2sIdx'] = len(c2List) # Find the index of target word.
                    meet = True
                else:
                    if wordFilter.match(w[0]):
                        if meet:
                            c2List.append(d['w2']) # Not lemmatize the main word in context
                            meet = False
                        elif len(w[1]) > 0:
                            if w[1][0] == 'N':
                                c2List.append(le.lemmatize(w[0], 'n').lower())
                            elif w[1][0] == 'J':
                                c2List.append(le.lemmatize(w[0], 'a').lower())
                            elif w[1][0] == 'V':
                                c2List.append(le.lemmatize(w[0], 'v').lower())
                            elif w[1][0] == 'R':
                                c2List.append(le.lemmatize(w[0], 'r').lower())
                            else:
                                c2List.append(le.lemmatize(w[0], 'v').lower())
                        else:
                            c2List.append(le.lemmatize(w[0], 'v').lower())
                    else:
                        pass
                        # print(w[0], 'Ignored')
            if c2List[d['w2sIdx']] != d['w2']:
                print(c2List[d['w2sIdx']], d['w2'])
            assert(c2List[d['w2sIdx']] == d['w2'])


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
    # processText8()
