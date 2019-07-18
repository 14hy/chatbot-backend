import config
from src.data.preprocessor import PreProcessor
from src.data.query import QueryMaker
from src.model.serving import TensorServer
from src.utils import Singleton
from src.db.queries import index as queries


class Handler(metaclass=Singleton):

    def __init__(self):
        self.CONFIG = config.HANDLER
        self.query_maker = QueryMaker()
        self.preprocessor = PreProcessor()
        self._model_wrapper = TensorServer()

    @staticmethod
    def get_response(answer, morphs, distance, measurement, text, sentiment):

        return {"morphs": morphs,  # 형태소 분석 된 결과
                "measurement": measurement,  # 유사도 측정의 방법, [jaccard, manhattan]
                "with": text,
                "distance": distance,  # 위 유사도의 거리
                "answer": answer,
                'sentiment': sentiment}

    def handle(self, chat, added_time=None):
        query = self.query_maker.make_query(chat=chat, added_time=added_time)
        if query.manhattan_similarity:
            distance = query.manhattan_similarity
        else:
            distance = query.jaccard_similarity
        queries.insert(query)

        # sentiment_score = self._model_wrapper.sentiment(chat=chat)[0]

        return self.get_response(answer=query.answer,
                                 morphs=query.morphs,
                                 distance=distance,
                                 measurement=query.measurement,
                                 text=query.matched_question,
                                 sentiment=None)
