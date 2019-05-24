from flask_restplus import Resource

from api.v2.visualize import api
from src.services.analysis import visualize_similarity


@api.route('/scatter/tsne/answer/<string:text>')
class Scatter(Resource):

    @api.doc('데이터베이스 답변들의 T-SNE 벡터 시각화')
    def get(self, text):
        try:
            return visualize_similarity(text, mode=0)
        except Exception as err:
            print(err)
            return {'error': str(err)}


@api.route('/scatter/tsne/query/<string:text>')
class Scatter(Resource):

    @api.doc('데이터베이스 사용자질문들의 T-SNE 벡터 시각화')
    def get(self, text):
        try:
            return visualize_similarity(text, mode=1)
        except Exception as err:
            print(err)
            return {'error': str(err)}
