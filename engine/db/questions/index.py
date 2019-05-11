import pickle

import logging
from engine.data.question import QuestionMaker
from engine.db.index import *
from engine.db.questions.question import convert_to_question, convert_to_document

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
        raise Exception('feature feature_vector is None')

    document = convert_to_document(question)

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
        questions.append(question)

    return questions


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


def remove_by_text():
    pass


def rebase():
    cursor = _questions.find({})

    for document in cursor:
        id
        question = convert_to_question(document)
        backup = None
        try:
            question = _question_maker.create_question(text=question.text,
                                                       category=question.category,
                                                       answer=question.answer)
            backup = question
            _questions.delete_one({'text': question.text})
            insert(question)
            print('rebase: {}'.format(question.text))
        except Exception as err:
            print('rebase: ', err)
            print(document)
            if backup:
                insert(backup)
            return document
