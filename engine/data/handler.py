# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및
from collections import OrderedDict

import numpy as np

import config
from engine.data.preprocess import PreProcessor
from engine.data.query import Query
from engine.db.mongo import PymongoWrapper
from engine.model.bert import Model
from engine.utils import Singleton

DEFAULT_CONFIG = config.DEFAULT_CONFIG


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
        return self.bert_model.extract_feature_vector(input_feature)

    def calc_jaccard_with_questions(self, chat):

        question_list = self.pymongo_wrapper.get_question_list()
        if question_list == []:
            return 0.0

        chat_morphs = self.preprocessor.str_to_morphs(chat).split(' ')
        chat_len = len(chat_morphs)
        distance_dict = {}

        for question in question_list:
            question_morphs = question.morphs.split(' ')
            question_len = len(question_morphs)
            num_union = chat_len + question_len
            num_joint = 0
            for i in range(len(question_morphs)):
                for j in range(len(chat_morphs)):
                    if question_morphs[i] == chat_morphs[j]:
                        num_joint += 1
            distance_dict[question.text] = num_joint / (num_union - num_joint)

        distance_dict = OrderedDict(sorted(distance_dict.items(), key=lambda t: t[1], reverse=True))

        print('*' * 50)
        i = 0
        for item, distance in distance_dict.items():
            print(i, 'th item: ', item, '/ jaccard_distance: ', distance)
            i += 1
            if i == 5:
                break

        item = list(distance_dict.items())[0][0]
        distance = list(distance_dict.items())[0][1]

        if distance >= 0.4:
            # TODO 임계값 설정으로 넘기기
            matched_question = self.pymongo_wrapper.get_question_by_text(item)
            return matched_question, distance
        else:
            return None, None


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
            if i == 5:
                break

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
        self.pymongo_wrapper = PymongoWrapper()

    def handle_chat(self, chat):
        '''
        :param chat: str
        :return: Query object
        '''
        keywords = self.query_maker.get_keywords(chat)
        matched_question, jaccard_score = self.query_maker.calc_jaccard_with_questions(chat)
        feature_vector = None
        feature_distance = None


        if jaccard_score is None:
            # 자카드 유사도가 임계값을 넘었을 시, 사용
            feature_vector = self.query_maker.get_feature_vector(chat)
            matched_question, feature_distance = self.query_maker.match_query_with_question(chat,
                                                                                feature_vector,
                                                                                keywords)

        query = Query(chat=chat,
                      feature_vector=feature_vector,
                      keywords=keywords,
                      matched_question=matched_question,
                      feature_distance=feature_distance,
                      jaccard_distance=jaccard_score)

        self.pymongo_wrapper.insert_query(query)

        return query


if __name__ == '__main__':
    ch = ChatHandler()
    ch.handle_chat('셔틀')