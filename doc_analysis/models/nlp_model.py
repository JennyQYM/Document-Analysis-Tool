# load the package
import os
import re

import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import sent_tokenize

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from rake_nltk import Rake

from .data_clean import DataClean

from wordcloud import WordCloud

import matplotlib
matplotlib.use('Agg')

import seaborn as sns

cleandata = DataClean()

class NLPModel:




    # model 1: wordcloud model

    def draw_wordcloud(self, doc_id, result):
        bag = pd.DataFrame(result.groupby(['word','doc_id'])['sent_frequency'].sum().reset_index())
        bag.index = bag.index.get_level_values(0)
        bag = bag[bag.doc_id==doc_id][['word', 'sent_frequency']]
        d = {}
        for a, x in bag.values:
            d[a] = x

        wordcloud = WordCloud()
        wordcloud.generate_from_frequencies(frequencies=d)
        wordcloud.to_file('static/wordcloud/wordcloud.png')


    # model 2.1: key words extraction

    def key_words_extraction(self, doc_id, topk, raw_dict):
        train = [cleandata.preprocessing(doc) for doc in raw_dict.values()]

        tf_vec = TfidfVectorizer()
        train_features = tf_vec.fit_transform(list(train))
        train_features.shape

        weight = train_features.toarray()
        words = tf_vec.get_feature_names()

        k_words = []
        for (docid, w) in zip(raw_dict.keys(), weight):
            if docid == doc_id:
                # order
                loc = np.argsort(-w)
                for i in range(topk):
                    k_words.append(words[loc[i]])
        return k_words


    # model 2.2: key phrases extraction

    def key_phrases_extraction(self, doc_id, raw_dict):
        # Uses stopwords for english from NLTK, and all puntuation characters.
        r = Rake()

        r.extract_keywords_from_text(raw_dict[doc_id])

        #r.get_ranked_phrases_with_scores()[:10]
        return r.get_ranked_phrases()[:5]


    # model 3: topic model

    def topic_word(self, raw_dict, k_topic, doc_id):
        docs_order_id = list(raw_dict.keys())
        corp_list = [cleandata.preprocessing(doc) for doc in raw_dict.values()]

        vectorizer = CountVectorizer()
        data_vectorized = vectorizer.fit_transform(corp_list)

        lda_model = LatentDirichletAllocation(n_topics=k_topic, random_state=2019)
        lda_top = lda_model.fit_transform(data_vectorized)

        domin_topic = lda_top[docs_order_id.index(doc_id)].argmax()
        domin_topic_prob = lda_top[docs_order_id.index(doc_id)][domin_topic]

        vocab = vectorizer.get_feature_names()
        comp = lda_model.components_[domin_topic, :]
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
        sorted_words = [x[0] for x in sorted_words]

        return sorted_words#, domin_topic, domin_topic_prob


    # model 4: topic model correlation

    def corr_doc(self, raw_dict, k_topic):
        docs_order_id = list(raw_dict.keys())
        corp_list = [cleandata.preprocessing(doc) for doc in raw_dict.values()]

        vectorizer = CountVectorizer()
        data_vectorized = vectorizer.fit_transform(corp_list)

        lda_model = LatentDirichletAllocation(n_topics=k_topic, random_state=2019)
        lda_top = lda_model.fit_transform(data_vectorized)

        corr_matrix = np.corrcoef(lda_top)
        corr_matrix_df = pd.DataFrame(corr_matrix, index=raw_dict.keys(), columns=raw_dict.keys())
        sns.heatmap(corr_matrix_df, cmap='coolwarm').get_figure().savefig("static/heatmap/heatmap.png")
