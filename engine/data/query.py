from collections import OrderedDict

import config
from engine.data.preprocess import PreProcessor
from engine.db.queries.query import Query
from engine.model.bert import Model

from engine.db.queries import index as queries
from engine.db.questions import index as questions


class QueryMaker():

    def __init__(self):

        self.preprocessor = PreProcessor()
        self.bert_model = Model()

        self.CONFIG = config.HANDLER

    def make_query(self, chat):
        def get_top(distances, top=1, threshold=0.5):
            assert type(distances) is OrderedDict
            output = {}

            for n, each in enumerate(list(distances.items())):
                item = each[0]
                distance = each[1]
                if distance >= threshold:
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

        queries.insert(query)

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

    def get_feature_vector(self, text):
        input_feature = self.preprocessor.create_InputFeature(text)
        return self.bert_model.extract_feature_vector(input_feature)
