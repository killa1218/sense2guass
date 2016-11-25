#!/usr/bin/python3
# -*- coding:utf-8 -*-

from nltk.corpus import wordnet as wn
import re
import json
import sys

class Dataset:
    """docstring for Dataset
        DataFormat:
        {
            (str-lemma, str-pos): [
                (str-lemma, str-pos),
                ...
                (str-lemma, str-pos)
            ],
            ...
            (str-lemma, str-pos): [
                (str-lemma, str-pos),
                ...
                (str-lemma, str-pos)
            ]
        }
    """

    # FORMAT_RE = re.compile(r'^\{\s*(\(\s*\w+\s*\,\s*\w+\s*\)\s*:\s*\[     \],)       *\s*\}$')

    def __init__(self, data):
        if isinstance(data, str):
            self._data = {}
        elif isinstance(data, dict):
            self._data = data
        else:
            raise 'PARAMETER TYPE ILLEGAL!'

    def get_size(self):
        return len(self._data.keys())

    # def get_hyponyms_list(self, word):
    #     if isinstance(self._data[word], list):

    #         return self._data[word]
    #     else:
    #         return []

    def test(self, store_path):
        fp = None

        if isinstance(store_path, str):
            fp = open(str, 'w')
        else:
            fp = open('./test_result', 'w')

        for hypernym, hyponym_list in self._data:
            hypernym_synsets_list = wn.synsets(hypernym[0], pos = hypernym[1])

            for hyponym in hyponym_list:
                hyponym_synsets_list = wn.synsets(hyponym[0], pos = hyponym[1])
                hightest_score = sys.float_info.max

                for d




# print(wn.lemma('salt.n.01').synset());
# print(wn.synset('walk.v.1').root_hypernyms());
# print(wn.synset('walk.v.01').entailments());
# print(wn.synset('walk.v.01').entailments());

# dic = wn.words();
# for key in dic:
#     print(key);
# for key in wn.all_lemma_names():
#     print(key);
print(wn.lemma(wn.morphy('vertices')));
print(wn.morphy('vertices'));
print(wn.synset('water.v.1').definition());
print(wn.synset('water.v.1').hyponyms());
print(wn.synset('water.v.1').hypernyms()[0].definition());
print(wn.synset('phone.n.1').hypernym_paths());

print(wn);
