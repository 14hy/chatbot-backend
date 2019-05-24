from flask_restplus import Resource

from api.v2.visualize import api
from src.services.analysis import get_MostCommonKeywords


@api.route('/bar/keyword/answer/<int:k>')
class Bar(Resource):

    @api.doc('데이터베이스 답변들의 키워드 TOP k 통계')
    def get(self, k):
        return get_MostCommonKeywords(n=k, mode=0)


@api.route('/bar/keyword/query/<int:k>')
class Bar(Resource):

    @api.doc('데이터베이스 사용자질문들의 키워드 TOP k 통계')
    def get(self, k):
        return get_MostCommonKeywords(n=k, mode=1)
