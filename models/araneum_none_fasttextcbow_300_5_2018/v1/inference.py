from arrested import Resource, Endpoint, json

from gensim.similarities import WmdSimilarity
import numpy as np

from .utils import EmbeddingWrapper

inference_endpoint_resource = \
    Resource('araneum_none_fasttextcbow_300_5_2018',
             __name__,
             url_prefix='/araneum_none_fasttextcbow_300_5_2018/v1')

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


class EndpointWmdSimilarity(Endpoint):

    name = 'wmdsimilarity'
    many = False
    url = '/wmdsimilarity'

    def post(self, *args, **kwargs):
        request = self.get_request_handler()
        data = request.process().data
        num_best = data.get("num_best") or 5
        corpus = data.get("corpus")
        query = data.get("query")

        wmd = WmdSimilarity(corpus, embedding.fasttext_model, num_best=num_best)
        # TODO: wmd result index has np.int64s and they are converted to float (need manual convert to int)
        result = np.asarray(wmd[query]).tolist()
        return json.dumps(result)


inference_endpoint_resource.add_endpoint(EndpointOne)
inference_endpoint_resource.add_endpoint(EndpointMany)
inference_endpoint_resource.add_endpoint(EndpointWmdSimilarity)
