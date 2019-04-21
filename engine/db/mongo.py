import numpy as np
import pickle
from pymongo import MongoClient

import config
from engine.data.query import Query
from engine.data.question import Question, QuestionMaker
from engine.utils import Singleton






class PymongoWrapper(metaclass=Singleton):
    def __init__(self):

        self.MONGODB_CONFIG = config.MONGODB_CONFIG

        _client = MongoClient(self.MONGODB_CONFIG['local_ip'], self.MONGODB_CONFIG['port'])
        self._db = _client[self.MONGODB_CONFIG['db_name']]
        self._questions = self._db[self.MONGODB_CONFIG['col_questions']]
        self._queries = self._db[self.MONGODB_CONFIG['col_queries']]
        self._question_maker = QuestionMaker()

    def _convert_to_question(self, document):
        feature_vector = pickle.loads(np.array(document['feature_vector']))
        question = Question(document['text'],
                            document['category'],
                            document['answer'],
                            feature_vector,
                            document['keywords'],
                            document['_id'])
        return question

    def _convert_to_query(self, document):
        feature_vector = pickle.loads(np.array(document['feature_vector']))
        matched_question = self.get_question_by_id(document['matched_question'])
        query = Query(document['chat'], feature_vector, document['keywords'],
                      matched_question, document['distance'])
        return query

    def create_question_and_insert(self, text, answer=None):
        question = self._question_maker.create_question(text, answer=answer)
        self.insert_question(question)
        return self

    def insert_question(self, question):
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
                    'keywords': question.keywords}

        return self._questions.update_one({'text': document['text']}, {'$set': document}, upsert=True) # update_one -> 중복 삽입을 막기 위해

    def read_from_txt(self, txt='../data/faq.txt'):
        '''텍스트 파일로 부터 질문을 읽어서 데이터 베이스에 저장,
        질문
        형식으로 저장
        '''
        with open(txt, mode='r', encoding='utf8') as f:
            for line in f:
                tokens = line.strip('\n')
                q = self._question_maker.create_question(tokens)
                self.insert_question(q)
        return self

    def remove_all_questions(self):
        '''questions collection의 모든 데이터를 삭제'''
        list = self.get_question_list()

        for each in list:
            _id = each.object_id
            print('삭제: ', each.text)
            self._questions.delete_one({'_id': _id})
        return self

    def get_question_list(self):
        '''

        :return: list of Question objects
        '''
        questions = []
        cursor = self._questions.find({})

        for document in cursor:
            question = self._convert_to_question(document)
            question.object_id = document['_id']
            questions.append(question)
        return questions

    def get_question_by_id(self, _id):
        '''

        :param _id: _id
        :return: Question object
        '''
        document = self._questions.find_one({'_id': _id})
        return self._convert_to_question(document)

    def get_question_by_text(self, text):
        '''

        :param text:
        :return: Question object
        '''
        document = self._questions.find_one({'text': text})
        return self._convert_to_question(document)

    def get_questions_by_keywords(self, keywords):
        '''

        :param keywords: list, keywords
        :return: list, of Question object
        '''
        output = []

        cursor = self._questions.find({'keywords': {'$in': keywords}})

        for document in cursor:
            question = self._convert_to_question(document)
            question.object_id = document['_id']
            output.append(question)
        return output

    def get_questions_by_category(self, category):
        '''
        :param category:
        :return: list of Question objects
        '''

        questions = []
        cursor = self._questions.find({'category': category})

        for document in cursor:
            question = self._convert_to_question(document)
            questions.append(question)

        return questions

    def delete_question(self, id):
        self._questions.delete_one(({'_id': id}))

    def replace_question(self, question):
        pass  # TODO

    def insert_query(self, query):

        feature_vector = pickle.dumps(query.feature_vector)

        document = {
            'chat': query.chat,
            'feature_vector': feature_vector,
            'keywords': query.keywords,
            'matched_question': query.matched_question.object_id,
            # 저장은 object id로 하지만 query 객체는 question 객체 이므로 헷갈리지 말 것
            'distance': query.distance
        }
        return self._queries.update_one({'chat': document['chat']}, {'$set': document}, upsert=True)

    def get_queries_list(self):
        queries = []
        cursor = self._queries.find({})

        for document in cursor:
            query = self._convert_to_query(document)
            queries.append(query)
        return queries

    def remove_all_queries(self):
        pass

if __name__ == '__main__':
    pw = PymongoWrapper()
