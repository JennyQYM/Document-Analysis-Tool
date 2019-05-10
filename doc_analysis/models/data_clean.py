# load the package
import os
import re
import pandas as pd
import numpy as np
import nltk
import string
import multiprocessing
import math
from itertools import repeat, chain

# LOAD FILE
# extract the document id

class DataClean:

    def docs_dict(self, path):
        """
        input: the path of test docs
        output: raw_docs dictionary
                the key of dictionary is doc id
                the value of dictionary is doc content
        """
        pattern = re.compile(r'\d+')
        docs_dict = {}
        for i in os.listdir(path):
            if i.endswith('.txt'):
                docs_dict[int(pattern.findall(i)[0])] = open(path+i, 'r', encoding='utf-8').read()
        return docs_dict



    # clean the doc
    def preprocessing(self, doc):
        """
        input: text
        output: cleaned lower case text without punctuation
        """
        # get the vocabulary frequency data frame

        # if want to check the stop words frequency
        # set stopwords = []
        stopwords = nltk.corpus.stopwords.words('english')
        # remove the punctuations
        transtbl = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        doc = doc.replace('\n', '')  # Remove \n
        doc = doc.translate(transtbl)  # Remove punctuation

        # Get tokens and unified the case
        tokens = [t.lower() for t in nltk.word_tokenize(doc) \
                                    if t.lower() not in stopwords and len(t) != 1]

        return ' '.join(tokens)


    # get the vocabulary frequency
    def voca_df(self, raw_docs):
        """
        input: text list
        output: vocabulary frequency dataframe
        """
        corps = ' '.join([self.preprocessing(doc) for doc in raw_docs])
        all_words = [w for w in corps.split()]
        voca = nltk.FreqDist(all_words)

        return pd.DataFrame(list(voca.items()), columns=['word', 'word_frequency'])


    # built the data frame for analyzing

    def ana_df(self, doc_id, doc_content):
        sent_orig = [sent.strip() for sent in nltk.sent_tokenize(doc_content)]
        sent_clean = [nltk.word_tokenize(self.preprocessing(sent)) for sent in sent_orig]
        d = {'doc_id': np.repeat(doc_id, len(sent_orig)),
             'sent_id': list(range(1,len(sent_orig)+1)),
             'sent_orig': sent_orig,
             'token_clean': sent_clean}
        return pd.DataFrame(d)

    # the worker to extract the target data frame
    def worker(self, w_list, analyze):

        temp = []
        for w in w_list:
            w_loc = [True if w in x else False for x in analyze.token_clean]
            groupdf = pd.DataFrame(analyze.loc[w_loc,['doc_id','sent_orig']] \
                        .groupby('doc_id')['sent_orig'].apply(list))
            groupdf['word'] = w
            groupdf['sent_frequency'] = groupdf.sent_orig.apply(len)
            temp.append(groupdf)
        return temp


    # split the word list to multiprocessing
    def split_list(self, lst_long,n):
        lst_splitted = []
        totalBatches = math.ceil(len(lst_long) / n)
        for i in range(totalBatches):
            lst_short = lst_long[i*n:(i+1)*n]
            lst_splitted.append(lst_short)
        return lst_splitted


    def creat_stat_df(self, word_list, analyze):
        p = multiprocessing.Pool(5)

        whole_df = []
        for i in self.split_list(word_list,100):
            lst_temp_dic = p.starmap(self.worker, zip(self.split_list(i,20), repeat(analyze)))
            for j in lst_temp_dic:
                whole_df.append(j)

        word_stat_df = pd.concat(list(chain.from_iterable(whole_df))).reset_index()
        return word_stat_df


    def run(self, path):
        # raw dictionary
        raw_dict = self.docs_dict(path)

        # vocabulary data frame
        voca = self.voca_df(raw_dict.values())

        # analysis data frame
        ana_df_list = []
        for doc_id, doc_content in raw_dict.items():
            ana_df_list.append(self.ana_df(doc_id, doc_content))
        # union all doc files
        analyze = pd.concat(ana_df_list)

        # word statistical data frame
        word_stat_df = self.creat_stat_df(voca['word'], analyze)

        # final result
        result = pd.merge(voca, word_stat_df, on='word')

        return raw_dict, result, voca
