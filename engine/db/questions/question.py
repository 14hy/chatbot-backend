import pickle

import numpy as np


class Question(object):
    def __init__(self, text, category, answer, feature_vector,
                 keywords, morphs, object_id=None):
        self.keywords = keywords
        self.object_id = object_id
        self.category = category
        self.answer = answer
        self.text = text
        self.feature_vector = feature_vector
        self.morphs = morphs


def convert_to_question(document):
    feature_vector = pickle.loads(np.array(document['feature_vector']))
    question = Question(document['text'],
                        document['category'],
                        document['answer'],
                        feature_vector,
                        document['keywords'],
                        document['morphs'],
                        document['_id'])
    return question
