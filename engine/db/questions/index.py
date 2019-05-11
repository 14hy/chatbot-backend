import pickle

from engine.data.question import QuestionMaker
from engine.db.index import *
from engine.db.questions.question import convert_to_question

_questions = db[MONGODB_CONFIG['col_questions']]
_question_maker = QuestionMaker()


def create_insert(text, answer=None, category=None):
    question = _question_maker.create_question(text, answer=answer, category=category)
    return insert(question)


def insert(question):
    '''
    :param question: Question object
    :return: InsertOneResult.inserted_id
    '''

    if question.feature_vector is None:
        raise Exception('feature vector is None')

    feature_vector = pickle.dumps(question.feature_vector)

    document = {'text': question.text,
                'answer': question.answer,
                'feature_vector': feature_vector,
                'category': question.category,
                'keywords': question.keywords,
                'morphs': question.morphs}

    return _questions.update_one({'text': document['text']}, {'$set': document},
                                 upsert=True)  # update_one -> 중복 삽입을 막기 위해


def find_all():
    '''

    :return: list of Question objects
    '''
    questions = []
    cursor = _questions.find({})

    for document in cursor:
        question = convert_to_question(document)
        question.object_id = document['_id']
        questions.append(question)

    return questions


def find_by_id(_id):
    '''

    :param _id: _id
    :return: Question object
    '''
    document = _questions.find_one({'_id': _id})
    return convert_to_question(document)


def find_by_text(text):
    '''

    :param text:
    :return: Question object
    '''
    document = _questions.find_one({'text': text})
    return convert_to_question(document)


def find_by_keywords(keywords):
    '''

    :param keywords: list, keywords
    :return: list, of Question object
    '''
    output = []

    cursor = _questions.find({'keywords': {'$in': keywords}})

    for document in cursor:
        question = convert_to_question(document)
        question.object_id = document['_id']
        output.append(question)
    return output


def find_by_category(category):
    '''
    :param category:
    :return: list of Question objects
    '''

    questions = []
    cursor = _questions.find({'category': category})

    for document in cursor:
        question = convert_to_question(document)
        questions.append(question)

    return questions


def delete(id):
    _questions.delete_one(({'_id': id}))
