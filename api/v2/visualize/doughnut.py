from flask_restplus import Resource

from api.v2.visualize import api
from src.services.analysis import visualize_category


@api.route('/doughnut/category/answer')
class Doughnut(Resource):

    @api.doc('데이터베이스 답변들의 카테고리 통계')
    def get(self):
        try:
            return visualize_category(mode=0)
        except Exception as err:
            print(err)
            return {'error': str(err)}


@api.route('/doughnut/category/query')
class Doughnut(Resource):

    @api.doc('데이터베이스 사용자질문들의 카테고리 통계')
    def get(self):
        try:
            return visualize_category(mode=1)
        except Exception as err:
            print(err)
            return {'error': str(err)}
