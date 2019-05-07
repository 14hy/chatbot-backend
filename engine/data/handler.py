# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및
from collections import OrderedDict
from pprint import pprint

import numpy as np

import config
from engine.data.preprocess import PreProcessor
from engine.data.query import QueryMaker
from engine.services.search import Search
from engine.services.shuttle import ShuttleBus
from engine.utils import Singleton


def cosine_similarity(a, b):
    '''
    성능이 좋지 않다. 모두 각도가 거의 비슷.
    :param a:
    :param b:
    :return:
    '''
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def manhattan_distance(a, b):
    '''
    :param a: sentence vector, [1, 768]
    :param b:
    :return:
    '''
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))
    return 1 + np.sum(np.abs(a - b))


def euclidean_distance(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))
    return 1 + np.linalg.norm(np.sqrt(np.dot((a - b), (a - b))))


class Handler(metaclass=Singleton):
    '''
    preprocess chat to query
    '''

    def __init__(self):
        self.query_maker = QueryMaker()
        self._service_shuttle = ShuttleBus()
        self._service_search = Search()
        self.preprocessor = PreProcessor()

    @staticmethod
    def create_answer(answer, morphs, distance, measurement):
        return {"morphs": morphs,  # 형태소 분석 된 결과
                "measurement": measurement,  # 유사도 측정의 방법, [jaccard, manhattan]
                "distance": distance,  # 위 유사도의 거리
                "answer": answer}

    def get_answer(self, chat):
        '''
        :param chat: str
        :return: Query object
        '''
        query = self.query_maker.make_query(chat)
        matched_question = query.matched_question
        morphs = self.preprocessor.get_morphs(chat)
        if not matched_question.answer:
            answer = self.answer_by_category(query)

        if query.jaccard_similarity:
            distance = query.jaccard_similarity
            measurement = 'jaccard_similiarity'
        elif query.manhattan_similarity:
            distance = query.manhattan_similarity
            measurement = 'manhattan_similarity'
        else:
            raise Exception('Query Distance가 모두 0')

        return self.create_answer(answer, morphs, distance, measurement)

    def answer_by_category(self, query):
        matched_question = query.matched_question
        category = matched_question.category

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk':
            return {"mode": "talk", "response": "Preparing for talk..."}
        elif category == 'food':
            return {'mode': 'food', 'response': 'Preparing for food...'}
        elif category == 'book':
            return {'mode': 'book', 'response': 'Taeuk will do'}
        elif category == 'search':
            response = self._service_search.response(query.chat)
            return {'mode': 'search', 'response': 'search result',
                    'context': '어디에서 찾았는지', 'confidence': '정확도'}


if __name__ == '__main__':
    ch = Handler()
    pprint(ch.get_answer('셔틀은 대체 언제 옵니까?'))
