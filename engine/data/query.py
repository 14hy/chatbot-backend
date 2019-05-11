from collections import OrderedDict
import numpy as np
from datetime import datetime, timezone
import config
from engine.data.preprocess import PreProcessor
from engine.db.queries.query import Query

from engine.db.questions import index as questions
from engine.model.serving import TensorServer
from engine.services.search import Search
from engine.services.shuttle import ShuttleBus

UTC = timezone.utc


def manhattan_distance(a, b):
    '''
    :param a: sentence feature_vector, [1, 768]
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
        self._service_shuttle = ShuttleBus()
        self._service_search = Search()

        self.CONFIG = config.QUERY

    def by_category(self, chat, category):

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk':
            return {"mode": "talk", "answer": "Preparing for talk..."}
        elif category == 'food':
            return {'mode': 'food', 'answer': 'Preparing for food...'}
        elif category == 'book':
            return {'mode': 'book', 'answer': 'Taeuk will do'}
        elif category == 'search':
            answer, context, tfidf_score = self._service_search.response(chat)
            if not answer:  # 정답이 오지 않았다면 일상대화로 유도
                return self.by_category(chat, category='talk')
            return {'mode': 'search', 'answer': answer,
                    'context': context, 'tfidf_score': tfidf_score}

    def make_query(self, chat, added_time=None):

        if not added_time:
            added_time = datetime.utcnow().astimezone(UTC)

        assert added_time.tzinfo is UTC

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
        jaccard_similarity = get_top(self.get_jaccard(chat), top=5)
        morphs = self.preprocessor.get_morphs(chat)

        feature_vector = None
        manhattan_similarity = None
        jaccard_similarity = None

        if not jaccard_similarity:
            feature_vector = self.modelWrapper.similarity(chat)
            feature_distances = self.get_similarity(feature_vector, keywords)
            top = get_top(feature_distances, top=5)
            matched_question, manhattan_similarity = get_one(top)
        else:
            matched_question, jaccard_similarity = get_one(jaccard_similarity)

        if jaccard_similarity:
            category = matched_question.category
            measurement = 'jaccard_similarity'
        elif manhattan_similarity:
            category = matched_question.category
            measurement = 'manhattan_similarity'
            if manhattan_similarity >= self.CONFIG['search_threshold']:
                category = 'search'
                matched_question.answer = None
        else:
            raise Exception('Query distance Error!')

        if not matched_question.answer:
            answer = self.by_category(chat, category)
        else:
            answer = matched_question.answer

        query = Query(chat=chat,
                      feature_vector=feature_vector,
                      keywords=keywords,
                      matched_question=matched_question,
                      manhattan_similarity=manhattan_similarity,
                      jaccard_similarity=jaccard_similarity,
                      added_time=added_time,
                      answer=answer,
                      morphs=morphs,
                      measurement=measurement,
                      category=category)

        return query

    def get_jaccard(self, chat):
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

    def get_similarity(self, feature_vector, keywords):
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
