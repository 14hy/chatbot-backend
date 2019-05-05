# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및
from collections import OrderedDict
from pprint import pprint

import numpy as np

import config
from engine.data.preprocess import PreProcessor
from engine.data.query import Query
from engine.db.mongo import PymongoWrapper
from engine.model.bert import Model
from engine.services.shuttle import ShuttleBus
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
            output = {}

            for n, each in enumerate(list(distances.items())):
                item = each[0]
                distance = each[1]
                if distance >= threshold:
                    question_matched = self.pymongo_wrapper.get_question_by_text(item)
                    output[n] = (question_matched, distance)

            if len(output) == 0:
                return None

            return output

        def get_one(top):
            return top[0][0], top[0][1]

        keywords = self.get_keywords(chat)
        jaccard_distances = get_top(self.get_jaccard_distances(chat), top=5)

        feature_vector = None
        manhattan_similarity = None
        jaccard_similarity = None
        if not jaccard_distances:
            feature_vector = self.get_feature_vector(chat)
            feature_distances = self.get_feature_distances(chat, feature_vector, keywords)
            top = get_top(feature_distances, top=5)
            matched_question, manhattan_similarity = get_one(top)
        else:
            matched_question, jaccard_similarity = get_one(jaccard_distances)

        query = Query(chat=chat,
                      feature_vector=feature_vector,
                      keywords=keywords,
                      matched_question=matched_question,
                      manhattan_similarity=manhattan_similarity,
                      jaccard_similarity=jaccard_similarity)

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
            return morphs['output'].split(' ')

        def _calc_jaacard(A, B):
            num_union = len(A) + len(B)
            num_joint = 0
            for a in A:
                for b in B:
                    if a == b:
                        num_joint += 1
            return num_joint / (num_union - num_joint)

        chat_morphs = _morphs_to_list(self.preprocessor.get_morphs(chat))

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
        self._service_shuttle = ShuttleBus()
        self.preprocessor = PreProcessor()

    def create_answer(self, answer, morphs, distance, measurement):
        return {"morphs": morphs,# 형태소 분석 된 결과
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
            answer = self.answer_by_category(matched_question)

        if query.jaccard_similarity:
            distance = query.jaccard_similarity
            measurement = 'jaccard_similiarity'
        elif query.manhattan_similarity:
            distance = query.manhattan_similarity
            measurement = 'manhattan_similarity'
        else:
            raise Exception('Query Distance가 모두 0')
        return self.create_answer(answer, morphs, distance, measurement)

    def answer_by_category(self, matched_question):

        category = matched_question.category

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk':
            return {"mode": "talk", "response": "Preparing for talk..."}
        elif category == 'food':
            return {'mode': 'food', 'response': 'Preparing for food...'}
        elif category == 'book':
            return {'mode': 'book', 'response': 'Taeuk will do'}


if __name__ == '__main__':
    ch = ChatHandler()
    pprint(ch.get_answer('셔틀은 대체 언제 옵니까?'))