#!/usr/bin/python3
# -*- coding:utf-8 -*-

from nltk.corpus import wordnet as wn
import re
import sys

class Dataset:
    """ docstring for Dataset
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
        if hasattr(self, '_size') and isinstance(self._size, int):
            pass
        else:
            self._size = 0

            for key, values in self._data:
                if isinstance(values, list):
                    self._size += len(values)

        return self._size


    # def get_hyponyms_list(self, word):
    #     if isinstance(self._data[word], list):

    #         return self._data[word]
    #     else:
    #         return []


    def get_dataset(self):
        return self._data

    def test(self, store_path):
        """ docstring for method test
            The output is stored in the file given by `store_path`, or by default ./test_result
        """
        fp = None

        if isinstance(store_path, str):
            fp = open(store_path, 'w')
        else:
            fp = open('./test_result', 'w')

        # try:
        for hypernym, hyponym_list in self._data.items():
            hypernym_synsets_list = wn.synsets(hypernym[0], pos = hypernym[1])

            if isinstance(hypernym_synsets_list, list):
                for hyponym in hyponym_list:
                    hyponym_synsets_list = wn.synsets(hyponym[0], pos = hyponym[1])

                    if isinstance(hyponym_synsets_list, list):
                        path_sim = sys.float_info.min
                        lch_sim = sys.float_info.min
                        wup_sim = sys.float_info.min
                        res_sim = sys.float_info.min
                        jcn_sim = sys.float_info.min
                        lin_sim = sys.float_info.min

                        for hypernym_synset in hypernym_synsets_list:
                            for hyponym_synset in hyponym_synsets_list:
                                hyponym_synset_s_hypernym_synsets = hyponym_synset.hypernyms()

                                for hyponym_synset_s_hypernym_synset in hyponym_synset_s_hypernym_synsets:

                                    tmp_sim = hypernym_synset.path_similarity(hyponym_synset_s_hypernym_synset)
                                    path_sim = tmp_sim if tmp_sim > path_sim else path_sim

                                    tmp_sim = hypernym_synset.lch_similarity(hyponym_synset_s_hypernym_synset)
                                    lch_sim = tmp_sim if tmp_sim > lch_sim else lch_sim

                                    tmp_sim = hypernym_synset.wup_similarity(hyponym_synset_s_hypernym_synset)
                                    wup_sim = tmp_sim if tmp_sim > wup_sim else wup_sim

                                    # tmp_sim = hypernym_synset.res_similarity(hyponym_synset_s_hypernym_synset)
                                    # res_sim = tmp_sim if tmp_sim > res_sim else res_sim

                                    # tmp_sim = hypernym_synset.jcn_similarity(hyponym_synset_s_hypernym_synset)
                                    # jcn_sim = tmp_sim if tmp_sim > jcn_sim else jcn_sim

                                    # tmp_sim = hypernym_synset.lin_similarity(hyponym_synset_s_hypernym_synset)
                                    # lin_sim = tmp_sim if tmp_sim > lin_sim else lin_sim

                        fp.write(hypernym[0] + '\t' + hyponym[0] + '\t' + str(path_sim) + '\t' + str(lch_sim) + '\t' + str(wup_sim) + '\t' + str(res_sim) + '\t' + str(jcn_sim) + '\t' + str(lin_sim) + '\n')
                    else:
                        print('[ERROR] Not found: ' + hyponym)

            else:
                print('[ERROR] Not found: ' + hypernym)
        # except BaseException:
        #     print(hypernym)


        fp.close()






# print(wn.lemma('salt.n.01').synset());
# print(wn.synset('walk.v.1').root_hypernyms());
# print(wn.synset('walk.v.01').entailments());
# print(wn.synset('walk.v.01').entailments());

# dic = wn.words();
# for key in dic:
#     print(key);
# for key in wn.all_lemma_names():
#     print(key);
# print(wn.lemma(wn.morphy('vertices')));
# print(wn.morphy('vertices'));
# print(wn.synset('water.v.1').definition());
# print(wn.synset('water.v.1').hyponyms());
# print(wn.synset('water.v.1').hypernyms()[0].definition());
# print(wn.synset('phone.n.1').hypernym_paths());

# print(wn);
