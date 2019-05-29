from collections import OrderedDict
import numpy as np
from datetime import datetime, timezone
import config
from src.data.preprocessor import PreProcessor
from src.data.question import QuestionMaker
from src.db.queries.query import Query
from src.db.questions import index as questions
from src.model.serving import TensorServer
from src.services.search import Search
from src.services.shuttle import ShuttleBus

UTC = timezone.utc

def cosine_similarity(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    a = np.reshape(a, newshape=(-1))
    b = np.reshape(b, newshape=(-1))

    i_p = np.dot(a, b)
    a_l1 = np.sqrt(np.dot(a, a))
    b_l1 = np.sqrt(np.dot(b, b))

    return i_p / (a_l1 * b_l1)

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


class QueryMaker(object):

    def __init__(self):

        self.preprocessor = PreProcessor()
        self.modelWrapper = TensorServer()
        self._question_maker = QuestionMaker()
        self._service_shuttle = ShuttleBus()
        self._service_search = Search()

        self.CONFIG = config.QUERY

    def by_category(self, chat, category, matched_question=None):

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk' or category == 'prepared':
            return {"mode": category, "answer": matched_question.answer}
        elif category == 'food':
            return {'mode': 'food', 'answer': '학식 보여주기'}
        elif category == 'book':
            return {'mode': 'book', 'answer': '도서관 모드 진입'}
        elif category == 'search':
            answer, output = self._service_search.response(chat)
            if not answer:  # 정답이 오지 않았다면 실패
                return {'mode': 'unknown', 'answer': '무슨 말인지 모르겠다냥~ 다시 해달라냥'}
            return {'mode': 'search',
                    'answer': answer,
                    'output': output}

    def make_query(self, chat, added_time=None, analysis=False):

        chat, removed = self.preprocessor.clean(chat)

        if chat is '' or chat is None:
            return None

        if not added_time:
            added_time = datetime.utcnow().astimezone(UTC)

        added_time.astimezone(UTC)

        def get_top(distances, measure='jaccard'):
            if not distances:
                return None
            assert type(distances) is OrderedDict
            output = {}

            for n, each in enumerate(list(distances.items())):
                item = each[0]
                distance = each[1]
                if distance >= self.CONFIG['jaccard_threshold'] and measure == 'jaccard':
                    question_matched = questions.find_by_text(item)
                    output[n] = (question_matched, distance)
                if distance >= self.CONFIG['cosine_threshold'] and measure == 'cosine':
                    question_matched = questions.find_by_text(item)
                    output[n] = (question_matched, distance)
                # question_matched = questions.find_by_text(item)
                # output[n] = (question_matched, distance)

            if len(output) == 0:
                return None

            return output

        feature_vector = self.modelWrapper.similarity(chat)
        jaccard_similarity = None
        top_feature_distance = None
        category = None
        keywords = self.preprocessor.get_keywords(chat)
        morphs = self.preprocessor.get_morphs(chat)

        # 우선 자카드 유사도 TOP 5를 찾음
        jaccard_top_distances = get_top(self.get_jaccard(chat), measure='jaccard')

        if jaccard_top_distances and not analysis:
            measurement = '자카드 유사도'
            matched_question, jaccard_similarity = jaccard_top_distances[0][0], jaccard_top_distances[0][1]
            category = matched_question.category

        else:  # 자카드 유사도가 없다면, 유클리드 또는 맨하탄 거리 비교로 넘어간다.
            feature_top_distances = get_top(self.get_similarity(chat, keywords, analysis), measure='cosine')
            if analysis:
                return feature_top_distances
            measurement = self.CONFIG['distance']
            if feature_top_distances is None:
                category = 'search'
                matched_question = None
                top_feature_distance = None
            else:
                matched_question = feature_top_distances[0][0]
                top_feature_distance = feature_top_distances[0][1]
                category = matched_question.category

        answer = self.by_category(chat, category, matched_question)

        query = Query(chat=chat,
                      feature_vector=feature_vector,
                      keywords=keywords,
                      matched_question=matched_question,
                      manhattan_similarity=top_feature_distance,
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

        def _calc_jaacard(A, B):
            A_output = A['text']
            B_output = B['text']
            VISITED = []
            num_union = len(A) + len(B) - 2  # output 뺀 것
            num_joint = 0
            for key_a, tag_a in A.items():
                for key_b, tag_b in B.items():
                    if key_a == 'text' or key_b == 'text':
                        continue
                    if key_a == key_b and tag_a == tag_b and key_a not in VISITED:
                        num_joint += 1
                        VISITED.append(key_a)
            return num_joint / (num_union - num_joint)

        chat_morphs = self.preprocessor.get_morphs(chat)

        for each in question_list:
            question_morphs = self.preprocessor.get_morphs(each.text)
            distance_dict[each.text] = _calc_jaacard(chat_morphs, question_morphs)

        return OrderedDict(sorted(distance_dict.items(), key=lambda t: t[1], reverse=True))

    def get_similarity(self, chat, keywords, analysis=False):
        assert chat is not None

        feature_vector = self.modelWrapper.similarity(chat)
        question_list = questions.find_by_keywords(keywords=keywords)
        if not question_list:  # 걸리는 키워드가 없는 경우 모두 다 비교 # search 로 넘어가는 것이, 성능적으로 좋을 듯
            # question_list = questions.find_all()
            return None
        # question_list = questions.find_all()

        distances = {}
        a_vector = self.get_weighted_average_vector(chat, feature_vector)
        if type(a_vector) != np.ndarray:
            return None

        for question in question_list:
            b_vector = self.get_weighted_average_vector(question.text, question.feature_vector)

            if self.CONFIG['distance'] == 'manhattan':
                distance = manhattan_distance(a_vector, b_vector)
            elif self.CONFIG['distance'] == 'euclidean':
                distance = euclidean_distance(a_vector, b_vector)
            elif self.CONFIG['distance'] == 'cosine':
                distance = cosine_similarity(a_vector, b_vector)
            else:
                raise Exception('CONFIG distance  measurement Error!')
            distances[question.text] = distance

        return OrderedDict(sorted(distances.items(), key=lambda t: t[1], reverse=True))  # 유클리드 할거면 바꿔야되

    def get_weighted_average_vector(self, text, vector):
        if len(vector.shape) == 1:
            return vector
        assert len(vector.shape) == 2

        text, _ = self.preprocessor.clean(text)
        tokens = self.preprocessor.str_to_tokens(text)

        idf_ = self._question_maker.idf_
        vocabulary_ = self._question_maker.vocabulary_
        output_vector = []

        for i, token in enumerate(tokens):

            idx = vocabulary_[token]
            idf = idf_[idx]
            # if token == '[UNK]':
            #     continue
            # elif idf == 1.0:
            #     output_vector.append(vector[i])
            #     continue
            # else:
            vector[i] += vector[i] * idf * self.CONFIG['idf_weight']
            output_vector.append(vector[i])

        if output_vector:
            output_vector = np.sum(output_vector, axis=0)
            return output_vector
        else:
            return np.array([0.0] * 768)


if __name__ == "__main__":
    test = QueryMaker()
