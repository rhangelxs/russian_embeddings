from arrested import Resource, Endpoint, json

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


inference_endpoint_resource.add_endpoint(EndpointOne)
inference_endpoint_resource.add_endpoint(EndpointMany)
