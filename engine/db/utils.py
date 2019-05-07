import pickle

import numpy as np

from engine.db.questions.question import Question


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

