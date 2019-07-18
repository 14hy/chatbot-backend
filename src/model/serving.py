import json

import numpy as np
import requests

import config
from src.utils import Singleton
from src.data.preprocessor import PreProcessor
"""
Docker Tensor-server 요청/ 응답
"""


class TensorServer(metaclass=Singleton):
    def __init__(self):
        self.preprocessor = PreProcessor()
        self.CONFIG = config.TENSOR_SERVING
        # search_v = json.loads(requests.get(self.CONFIG['url-search-v']).text)
        # sentiment_v = json.loads(requests.get(self.CONFIG['url-sentiment-v']).text)
        # similarity_v = json.loads(requests.get(self.CONFIG['url-similarity-v']).text)
        # print('TensorServer Running')
        # print('QA - {}'.format(search_v))
        # print('Sentiment - {}'.format(sentiment_v))
        # print('Similarity - {}'.format(similarity_v))

    @staticmethod
    def create_request(features):
        request_json = {
            'instances': [
                {
                    'input_ids': features.input_ids,
                    'input_masks': features.input_masks,
                    'segment_ids': features.segment_ids
                }
            ]
        }
        return request_json

    def sentiment(self, chat):
        chat, _ = self.preprocessor.clean(chat=chat)
        features = self.preprocessor.create_InputFeature(query_text=chat)
        response = requests.post(self.CONFIG['url-sentiment'], json=self.create_request(features))
        predict = json.loads(response.text)['predictions'][0]
        return predict

    def similarity(self, chat):
        chat, _ = self.preprocessor.clean(chat=chat)
        features = self.preprocessor.create_InputFeature(query_text=chat)
        _length = np.sum(features.input_masks)

        response = requests.post(self.CONFIG['url-similarity'], json=self.create_request(features))
        response = json.loads(response.text)
        similarity_vector = response['predictions'][0]
        similarity_vector = np.array(similarity_vector)[1:_length - 1]

        return similarity_vector

    def search(self, chat, context):
        chat, _ = self.preprocessor.clean(chat=chat)
        features = self.preprocessor.create_InputFeature(chat, context)

        response = requests.post(self.CONFIG['url-search'], json=self.create_request(features))
        response = json.loads(response.text)

        start = response['predictions'][0]['start_pred']
        end = response['predictions'][0]['end_pred']

        start = np.argmax(start, axis=-1)
        end = np.argmax(end, axis=-1)
        return self.preprocessor.idx_to_orig(start, end, features)


if __name__ == '__main__':
    test = TensorServer()
