import config
from engine.data.preprocess import PreProcessor
from engine.data.query import QueryMaker
from engine.services.search import Search
from engine.services.shuttle import ShuttleBus
from engine.utils import Singleton
from engine.db.queries import index as queries



class Handler(metaclass=Singleton):

    def __init__(self):
        self.CONFIG = config.HANDLER
        self.query_maker = QueryMaker()
        self._service_shuttle = ShuttleBus()
        self._service_search = Search()
        self.preprocessor = PreProcessor()

    @staticmethod
    def get_response(answer, morphs, distance, measurement):
        return {"morphs": morphs,  # 형태소 분석 된 결과
                "measurement": measurement,  # 유사도 측정의 방법, [jaccard, manhattan]
                "distance": distance,  # 위 유사도의 거리
                "answer": answer}

    def handle(self, chat):
        query = self.query_maker.make_query(chat)
        matched_question = query.matched_question
        morphs = self.preprocessor.get_morphs(chat)

        if query.jaccard_similarity:
            distance = query.jaccard_similarity
            measurement = 'jaccard_similarity'
            query.category = matched_question.category
        elif query.manhattan_similarity:
            distance = query.manhattan_similarity
            measurement = 'manhattan_similarity'
            # query.category = matched_question.category
            # if distance >= self.CONFIG['search_threshold']:
            query.category = 'search'
            matched_question.answer = None  # FOR TEST ONLY
        else:
            raise Exception('Query distance Error!')

        if not matched_question.answer:
            answer = self.by_category(query)
        else:
            answer = matched_question.answer

        queries.insert(query)

        return self.get_response(answer, morphs, distance, measurement)

    def by_category(self, query):
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
