from arrested import Resource, Endpoint, json

import numpy as np
from functools import lru_cache

from .hashable_cache import hashable_cache
from .utils import EmbeddingWrapper


inference_endpoint_resource = \
    Resource('universal_sentence_encoder_multilingual_qa',
             __name__,
             url_prefix='/universal_sentence_encoder_multilingual_qa/v1')


embedding = EmbeddingWrapper()


class EndpointMany(Endpoint):
    name = 'list'
    many = True
    url = "/inference"

    def post(self, *args, **kwargs):
        request = self.get_request_handler()
        token = request.process().data.get("token")
        vector = embedding.str2vec(token)
        return json.dumps(vector.tolist())


class EndpointOne(Endpoint):

    name = 'object'
    many = False
    url = '/inference/<string:token>'

    def get(self, *args, **kwargs):
        token = self.kwargs.get('token')
        vector = embedding.str2vec(token)
        return json.dumps(vector.tolist())


@hashable_cache(lru_cache(maxsize=256))
def return_qa(corpus, num_best):
    # TODO: implement
    # return WmdSimilarity(corpus, embedding.fasttext_model, num_best=num_best)
    pass


class EndpointQA(Endpoint):
    # TODO: work as mock
    name = 'qa'
    many = False
    url = '/qa'

    def post(self, *args, **kwargs):
        request = self.get_request_handler()
        data = request.process().data
        num_best = data.get("num_best") or 5
        corpus = data.get("corpus")
        query = data.get("query")

        qa = return_qa(corpus, num_best)

        # TODO: wmd result index has np.int64s and they are converted to float (need manual convert to int)
        result = np.asarray(qa[query]).tolist()
        return json.dumps(result)


inference_endpoint_resource.add_endpoint(EndpointOne)
inference_endpoint_resource.add_endpoint(EndpointMany)
inference_endpoint_resource.add_endpoint(EndpointQA)
