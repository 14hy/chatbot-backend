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

    def make_query(self, chat):
        def get_top(distances, top=1, threshold=0.5):
            assert type(distances) is OrderedDict
            assert top >= 1

            top = {}

            for n in top:
                item = list(distances.items())[n][0]
                distance = list(distances.items())[n][1]
                if distance >= threshold:
                    question_matched = self.pymongo_wrapper.get_question_by_text(item)
                    top[n] = (question_matched, distance)

            if top == {}:
                return None

            return top

        def get_one(top, num):
            assert top == {}
            return top[0][0], top[0][1]
        keywords = self.get_keywords(chat)
        jaccard_distances = self.get_jaccard_distances(chat)

        feature_vector = None
        feature_distance = None
        jaccard_score = None

        if not jaccard_distances:
            feature_vector = self.query_maker.get_feature_vector(chat)
            feature_distances = self.get_feature_distances(chat, feature_vector, keywords)
            top = get_top(feature_distances, top=5)
            matched_question, feature_distance = get_one(top)
        else:
            top = get_top(jaccard_distances, top=5)
            matched_question, jaccard_score = get_one(top)

        query = Query(chat=chat,
                      feature_vector=feature_vector,
                      keywords=keywords,
                      matched_question=matched_question,
                      feature_distance=feature_distance,
                      jaccard_score=jaccard_score)

        self.pymongo_wrapper.insert_query(query)

        return query

    def get_keywords(self, text):
        return self.preprocessor.get_keywords(text)

    def get_feature_vector(self, text):
        input_feature = self.preprocessor.create_InputFeature(text)
        return self.bert_model.extract_feature_vector(input_feature)

    def get_jaccard_distances(self, chat):
        assert chat is not None
        question_list = self.pymongo_wrapper.get_question_list()
        assert question_list is not None

        distance_dict = {}

        def _morphs_to_list(morphs):
            return morphs.split(' ')

        def _calc_jaacard(A, B):
            num_union = len(A) + len(B)
            num_joint = 0
            for a in A:
                for b in B:
                    if a == b:
                        num_joint += 1
            return num_joint / (num_union - num_joint)

        chat_morphs = _morphs_to_list(self.preprocessor.str_to_morphs(chat))

        for question in question_list:
            question_morphs = _morphs_to_list(question.morphs)
            distance_dict[question.text] = _calc_jaacard(chat_morphs, question_morphs)

        return OrderedDict(sorted(distance_dict.items(), key=lambda t: t[1], reverse=True))


    def get_feature_distances(self, chat, feature_vector, keywords):
        '''
        :param chat:
        :param feature_vector:
        :param keywords:
        :return:
        '''
        assert feature_vector is not None

        question_list = self.pymongo_wrapper.get_questions_by_keywords(keywords=keywords)
        if not question_list: # 걸리는 키워드가 없는 경우 모두 다 비교
            question_list = self.pymongo_wrapper.get_question_list()

        distances = {}

        for question in question_list:
            a = feature_vector
            b = question.feature_vector
            if self.DEFAULT_CONFIG['distance'] == 'manhattan':
                distance = manhattan_distance(a, b)
            elif self.DEFAULT_CONFIG['distance'] == 'euclidean':
                distance = euclidean_distance(a, b)
            else:
                raise Exception('DEFAUL_CONFIG - distance 는 "euclidean", "manhattan" 둘중 하나 여야 합니다.')
            distances[question.text] = distance

        return OrderedDict(sorted(distances.items(), key=lambda t: t[1]))


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
        query = self.query_maker.make_query(chat)
        return query



if __name__ == '__main__':
    ch = ChatHandler()
    ch.handle_chat('셔틀')