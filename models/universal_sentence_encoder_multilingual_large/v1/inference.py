from arrested import Resource, Endpoint, json

import numpy as np
from functools import lru_cache

from .hashable_cache import hashable_cache
from .utils import EmbeddingWrapper


inference_endpoint_resource = \
    Resource('universal_sentence_encoder_multilingual_large',
             __name__,
             url_prefix='/universal_sentence_encoder_multilingual_large/v1')


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


inference_endpoint_resource.add_endpoint(EndpointOne)
inference_endpoint_resource.add_endpoint(EndpointMany)
