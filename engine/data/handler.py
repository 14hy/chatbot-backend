# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및
from collections import OrderedDict

import numpy as np

import config
from engine.data.preprocess import PreProcessor
from engine.db.mongo import PymongoWrapper
from engine.model.bert import Model
from engine.utils import Singleton

DEFAULT_CONFIG = config.DEFAULT_CONFIG


class Query(object):  # TODO 어떤 정보가 필요 할 지 계속 고민 해보기
    def __init__(self, chat, feature_vector, keywords, matched_question, distance):
        self.chat = chat
        self.feature_vector = feature_vector
        self.keywords = keywords
        self.matched_question = matched_question  # 어떤 질문과 매칭 되었었는지.
        self.distance = distance  # 거리는 어떠 하였는 지

    def __str__(self):
        return 'Query chat:%5s, keywords%10s, question:%5s, distance:%5.3f' \
               % (self.chat, self.keywords, self.matched_question.text, self.distance)


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


class QueryMaker():

    def __init__(self):
        self.pymongo_wrapper = PymongoWrapper()

        self.preprocessor = PreProcessor()
        self.bert_model = Model()

        self.DEFAULT_CONFIG = config.DEFAULT_CONFIG

    def get_keywords(self, text):
        return self.preprocessor.get_keywords(text)

    def get_feature_vector(self, text):
        input_feature = self.preprocessor.create_InputFeature(text)
        return self.bert_model.extract_feature_vector(input_feature,
                                                      DEFAULT_CONFIG['feature_layers'])

    def match_query_with_question(self, chat, feature_vector, keywords):
        '''
        :param chat:
        :param feature_vector:
        :param keywords:
        :return:
        '''
        assert feature_vector is not None

        question_list = self.pymongo_wrapper.get_questions_by_keywords(keywords=keywords)
        if question_list == []: # 걸리는 키워드가 없는 경우 모두 다 비교
            question_list = self.pymongo_wrapper.get_question_list()
        distance_dict = {}

        print('***주어진 query, "{}" 와의 거리 비교 테스트***'.format(chat))
        # print('*' * 50)
        for question in question_list:
            a = feature_vector
            b = question.feature_vector
            if self.DEFAULT_CONFIG['distance'] == 'manhattan':
                distance = manhattan_distance(a, b)
            elif self.DEFAULT_CONFIG['distance'] == 'euclidean':
                distance = euclidean_distance(a, b)
            else:
                raise Exception('DEFAUL_CONFIG - distance 는 "euclidean", "manhattan" 둘중 하나 여야 합니다.')
            distance_dict[question.text] = distance

        distance_dict = OrderedDict(sorted(distance_dict.items(), key=lambda t: t[1]))

        # LOGGING
        print('*' * 50)
        i = 0
        for item, distance in distance_dict.items():
            print(i, 'th item: ', item, '/ distance: ', distance)
            i += 1

        # output
        item = list(distance_dict.items())[0][0]
        distance = list(distance_dict.items())[0][1]
        matched_question = self.pymongo_wrapper.get_question_by_text(item)

        return matched_question, distance


class ChatHandler(metaclass=Singleton):
    '''
    preprocess chat to query
    '''

    def __init__(self):
        self.query_maker = QueryMaker()

    def create_query_from_chat(self, chat):
        '''
        :param chat: str
        :return: Query object
        '''
        keywords = self.query_maker.get_keywords(chat)
        feature_vector = self.query_maker.get_feature_vector(chat)
        matched_question, distance = self.query_maker.match_query_with_question(chat,
                                                                                feature_vector,
                                                                                keywords)

        query = Query(chat, feature_vector, keywords, matched_question, distance)
        return query


if __name__ == '__main__':
    ch = ChatHandler()
    print(ch.create_query_from_chat('셔틀 버스 언제 오나요?'))
    print(ch.create_query_from_chat('식당 메뉴 추천 해주세요'))
    print(ch.create_query_from_chat('점심 메뉴 추천'))
    print(ch.create_query_from_chat('학생 식당 메뉴는 무엇 입니까?'))
    print(ch.create_query_from_chat('학생 식당 메뉴 알려줘.'))
    print(ch.create_query_from_chat('강의평가를 어떻게 하냐?????.'))
    print(ch.create_query_from_chat('강의평가를 어떻게 해요?'))
