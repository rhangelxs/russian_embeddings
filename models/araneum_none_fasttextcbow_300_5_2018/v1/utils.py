import os

import gensim
import numpy

model_dir = os.path.dirname(os.path.dirname(__file__))


class EmbeddingWrapper:
    def __init__(self):
        fasttext_gensim_model = gensim.models.KeyedVectors.load(
            os.path.join(model_dir, 'araneum_none_fasttextcbow_300_5_2018.model')
        )
        self.fasttext_model = fasttext_gensim_model

    def str2vec(self, string):
        result = numpy.zeros([300])
        words = string.split()
        empty_counter = 0
        for word in words:
            try:
                wv_normed = self.fasttext_model[word]
                result += wv_normed
            except KeyError:
                empty_counter += 1
                pass
        nonempty_len = len(words)-empty_counter
        if nonempty_len:
            result = result/nonempty_len
        return result
