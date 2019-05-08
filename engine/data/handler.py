# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및
from pprint import pprint

import config
from engine.data.preprocess import PreProcessor
from engine.data.query import QueryMaker
from engine.services.search import Search
from engine.services.shuttle import ShuttleBus
from engine.utils import Singleton
from engine.db.queries import index as queries


class Handler(metaclass=Singleton):
    '''
    preprocess chat to query
    '''

    def __init__(self):
        self.CONFIG = config.HANDLER
        self.query_maker = QueryMaker()
        self._service_shuttle = ShuttleBus()
        self._service_search = Search()
        self.preprocessor = PreProcessor()

    @staticmethod
    def create_answer(answer, morphs, distance, measurement):
        return {"morphs": morphs,  # 형태소 분석 된 결과
                "measurement": measurement,  # 유사도 측정의 방법, [jaccard, manhattan]
                "distance": distance,  # 위 유사도의 거리
                "answer": answer}

    def get_answer(self, chat):
        '''
        :param chat: str
        :return: Query object
        '''
        query = self.query_maker.make_query(chat)
        matched_question = query.matched_question
        morphs = self.preprocessor.get_morphs(chat)

        if query.jaccard_similarity:
            distance = query.jaccard_similarity
            measurement = 'jaccard_similiarity'
            query.category = matched_question.category
        elif query.manhattan_similarity:
            distance = query.manhattan_similarity
            measurement = 'manhattan_similarity'
            # query.category = matched_question.category
            # if distance >= self.CONFIG['search_threshold']:
            query.category = 'search'
        else:
            raise Exception('Query distance Error!')

        if not matched_question.answer:
            answer = self.answer_by_category(query)
        else:
            answer = matched_question.answer

        queries.insert(query)

        return self.create_answer(answer, morphs, distance, measurement)

    def answer_by_category(self, query):
        category = query.category

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk':
            return {"mode": "talk", "response": "Preparing for talk..."}
        elif category == 'food':
            return {'mode': 'food', 'response': 'Preparing for food...'}
        elif category == 'book':
            return {'mode': 'book', 'response': 'Taeuk will do'}
        elif category == 'search':
            response, context, tfidf_score = self._service_search.response(query.chat)
            return {'mode': 'search', 'response': response,
                    'context': context, 'tfidf_score': tfidf_score}


if __name__ == '__main__':
    ch = Handler()
    pprint(ch.get_answer('셔틀은 대체 언제 옵니까?'))
