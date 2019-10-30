import numpy

import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece


class EmbeddingWrapper:
    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"
        # Set up graph.
        g = tf.Graph()
        with g.as_default():
            self.module = hub.Module(module_url)  # load tfhub module
            self.question = tf.placeholder(dtype=tf.string, shape=[None])  # question
            self.response = tf.placeholder(dtype=tf.string, shape=[None])  # response
            self.response_context = tf.placeholder(
                dtype=tf.string, shape=[None])  # response context
            self.question_embedding = self.module(
                self.question, signature="question_encoder", as_dict=True)
            self.response_embedding = self.module(
                inputs={
                    "input": self.response,
                    "context": self.response_context
                },
                signature="response_encoder",
                as_dict=True)
            init_op = tf.group(
                [tf.global_variables_initializer(),
                 tf.tables_initializer()])
        g.finalize()

        # Initialize session.
        session = tf.Session(graph=g)
        session.run(init_op)
        self.session = session

    def str2vec(self, string):
        result = self.session.run(self.question_embedding, feed_dict={self.question: [string]})['outputs'][0]
        return result
