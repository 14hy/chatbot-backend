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
    def get_response(answer, morphs, distance, measurement, text):
        return {"morphs": morphs,  # 형태소 분석 된 결과
                "measurement": measurement,  # 유사도 측정의 방법, [jaccard, manhattan]
                "with": text,
                "distance": distance,  # 위 유사도의 거리
                "answer": answer}

    def handle(self, chat):
        query = self.query_maker.make_query(chat)
        matched_question = query.matched_question
        morphs = self.preprocessor.get_morphs(chat)
        text = None

        if query.jaccard_similarity:
            distance = query.jaccard_similarity
            measurement = 'jaccard_similarity'
            query.category = matched_question.category
            text = matched_question.text
        elif query.manhattan_similarity:
            distance = query.manhattan_similarity
            measurement = 'manhattan_similarity'
            query.category = matched_question.category
            if distance >= self.CONFIG['search_threshold']:
                query.category = 'search'
                matched_question.answer = None
            text = matched_question.text
        else:
            raise Exception('Query distance Error!')

        if not matched_question.answer:
            answer = self.by_category(query)
        else:
            answer = matched_question.answer

        queries.insert(query)

        return self.get_response(answer, morphs, distance, measurement, text)

    def by_category(self, query):
        category = query.category

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk':
            return {"mode": "talk", "answer": "Preparing for talk..."}
        elif category == 'food':
            return {'mode': 'food', 'answer': 'Preparing for food...'}
        elif category == 'book':
            return {'mode': 'book', 'answer': 'Taeuk will do'}
        elif category == 'search':
            answer, context, tfidf_score = self._service_search.response(query.chat)
            if not answer:  # 정답이 오지 않았다면 일상대화로 유도
                query.category = 'talk'
                return self.by_category(query)
            return {'mode': 'search', 'answer': answer,
                    'context': context, 'tfidf_score': tfidf_score}
