from collections import OrderedDict

import numpy as np

import config
from engine.data.preprocess import PreProcessor
from engine.db.queries.query import Query

from engine.db.questions import index as questions
from engine.model.serving import TensorServer


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

        self.preprocessor = PreProcessor()
        self.modelWrapper = TensorServer()

        self.CONFIG = config.QUERY

    def make_query(self, chat):
        def get_top(distances, top=1):
            assert type(distances) is OrderedDict
            output = {}

            for n, each in enumerate(list(distances.items())):
                item = each[0]
                distance = each[1]
                if distance >= self.CONFIG['jaccard_threshold']:
                    question_matched = questions.find_by_text(item)
                    output[n] = (question_matched, distance)

            if len(output) == 0:
                return None

            return output

        def get_one(top):
            return top[0][0], top[0][1]

        keywords = self.preprocessor.get_keywords(chat)
        jaccard_distances = get_top(self.get_jaccard_distances(chat), top=5)

        feature_vector = None
        manhattan_similarity = None
        jaccard_similarity = None
        if not jaccard_distances:
            feature_vector = self.modelWrapper.similarity(chat)
            feature_distances = self.get_feature_distances(feature_vector, keywords)
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

        return query

    def get_jaccard_distances(self, chat):
        assert chat is not None
        question_list = questions.find_all()
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

        for each in question_list:
            question_morphs = _morphs_to_list(each.morphs)
            distance_dict[each.text] = _calc_jaacard(chat_morphs, question_morphs)

        return OrderedDict(sorted(distance_dict.items(), key=lambda t: t[1], reverse=True))

    def get_feature_distances(self, feature_vector, keywords):
        assert feature_vector is not None

        question_list = questions.find_by_keywords(keywords=keywords)
        if not question_list:  # 걸리는 키워드가 없는 경우 모두 다 비교
            question_list = questions.find_all()

        distances = {}

        for question in question_list:
            a = feature_vector
            b = question.feature_vector
            if self.CONFIG['distance'] == 'manhattan':
                distance = manhattan_distance(a, b)
            elif self.CONFIG['distance'] == 'euclidean':
                distance = euclidean_distance(a, b)
            else:
                raise Exception('CONFIG distance  measurement Error!')
            distances[question.text] = distance

        return OrderedDict(sorted(distances.items(), key=lambda t: t[1]))
