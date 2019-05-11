import pickle

import numpy as np


class Question(object):
    def __init__(self, text, category, answer, feature_vector,
                 morphs, keywords=None):
        self.category = category
        self.answer = answer
        self.text = text
        self.feature_vector = feature_vector
        self.morphs = morphs
        self.keywords = keywords


def convert_to_question(document):
    vector = pickle.loads(np.array(document['feature_vector']))
    question = Question(text=document['text'],
                        category=document['category'],
                        answer=document['answer'],
                        feature_vector=vector,
                        morphs=document['morphs'],
                        keywords=document['keywords'])
    return question


def convert_to_document(question):
    feature_vector = pickle.dumps(question.feature_vector)
    document = {'text': question.text,
                'answer': question.answer,
                'feature_vector': feature_vector,
                'category': question.category,
                'keywords': question.keywords,
                'morphs': question.morphs}
    return document
