# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및
import numpy as np

import config
from engine.data.preprocess import PreProcessor
from engine.db.mongo import PymongoWrapper
from engine.model.bert import Model


DEFAULT_CONFIG = config.DEFAULT_CONFIG

class Query(object):
    def __init__(self, chat):

        self.response = None
        self.chat = chat
        self.category = None
        self.feature_vector = None


def cosine_similarity(a, b):
    '''
    성능이 좋지 않다. 모두 각도가 거의 비슷.
    :param a:
    :param b:
    :return:
    '''
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def manhattan_distance(a, b):
    '''
    :param a: sentence vector, [1, 768]
    :param b:
    :return:
    '''
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))
    return 1+np.sum(np.abs(a - b))


def euclidean_distance(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))
    return 1 + np.linalg.norm(np.sqrt(np.dot((a-b), (a-b))))


class Categorizer():
    '''

    '''
    def __init__(self):
        self._pymongo_wrapper = PymongoWrapper()

        self.preprocessor = PreProcessor()
        self.bert_model = Model()

    def categorize(self, query):
        '''
        유사도분석을 통해서 카테고리화 한다.
        :param query:
        :return:

        사전에 준비 된 질문과의 유사도를 비교
        비효율적이고 계산이 비싸기 때문에 충분한 데이터가 쌓이면
        classification model을 만들어서 대체하기
        혹은 사용자에게 1차 분류를 시킴으로써 계산량 감소 가능
        혹은 비슷한 질문을 클러스터링하여 n -> m개로 줄이기
        '''

        question_list = self._pymongo_wrapper.get_question_list()

        # 모든 질문에 대해, 거리를 비교하여
        input_feature = self.preprocessor.create_InputFeature(query.chat)
        feature_vector = self.bert_model.extract_feature_vector(input_feature,
                                                                DEFAULT_CONFIG['feature_layers'])
        query.feature_vector = feature_vector
        print('***주어진 query, "{}" 와의 거리 비교 테스트***'.format(query.chat))
        measure = 'manhattan'
        for question in question_list:
            distance = manhattan_distance(query.feature_vector, question.feature_vector)
            print('"{}" 와의 {}거리: {}'.format(question.text, measure, distance))
        measure = 'euclidean'
        for question in question_list:
            distance = euclidean_distance(query.feature_vector, question.feature_vector)
            print('"{}" 와의 {}거리: {}'.format(question.text, measure, distance))
        # measure = 'cosine'
        # for question in question_list:
        #     distance = cosine_similarity(query.feature_vector, question.feature_vector)
        #     print('"{}" 와의 {}거리: {}'.format(question.text, measure, distance))

        # 1. 코사인 거리 # TODO
        # 2. 맨하탄 거리 # TODO
        # 3. 유클리드 거리 # TODO
        # 4. 자카드 유사도 # TODO

        # 추출한 피쳐를 바탕으로 카테고리 추출

        return query

class ChatHandler():
    '''
    preprocess chat to query
    '''
    def __init__(self):
        self.categorizer = Categorizer()
        pass

    def create_query_from_chat(self, chat):
        '''
        :param chat: str
        :return: Query object
        '''
        query = Query(chat)

        return self.categorizer.categorize(query)

if __name__ == '__main__':
    ch = ChatHandler()
    ch.create_query_from_chat('점심뭐먹을까?')
    ch.create_query_from_chat('셔틀 버스 언제 오나요?')
    ch.create_query_from_chat('밥주세용밥?')