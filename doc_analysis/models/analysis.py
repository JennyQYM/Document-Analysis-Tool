# load the package
import os
from .data_clean import DataClean
from .nlp_model import NLPModel

class Result:

    def get_data(self, path):
        clean_data = DataClean()
        raw_dict, result, voca = clean_data.run(path)
        return raw_dict, result, voca

    def nlp_result(self, id, result, raw_dict):
        models = NLPModel()
        wc_image = models.draw_wordcloud(id, result)
        keywords = models.key_words_extraction(id, 10, raw_dict)
        keyphrases = models.key_phrases_extraction(id, raw_dict)
        topics = models.topic_word(raw_dict, 10, id)
        corr = models.corr_doc(raw_dict, 10)
        return wc_image, keywords, keyphrases, topics, corr
