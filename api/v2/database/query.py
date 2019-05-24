from flask_restplus import Resource, Namespace
from api.common.settings import *

QUESTIONS = 0
QUERIES = 1
api = Namespace('v2/db/query', 'DataBase Query endpoint')


@api.route('/')
class Query(Resource):

    def get(self):
        return {'status': '구현 중'}


@api.route('/_id/<string:string>')
@api.doc('해당 아이디에 해당하는 질문')
class Query(Resource):

    def get(self, string):
        return {'status': '구현 중'}

    def delete(self, string):
        return {'status': '구현 중'}

    def patch(self, string):
        return {'status': '구현 중'}


@api.route('/text/<string:string>')
class Query(Resource):

    @api.doc('해당 문자열이 들어간 질문 리스트')
    def get(self, string):
        return {'status': '구현 중'}


@api.route('/<string:keyword>')
class Query(Resource):

    @api.doc('해당 하는 키워드가 있는 쿼리의 아이디, 텍스트, 답변, 카테고리 리스트 + @?')
    def get(self, keyword):
        return {'status': '구현 중'}


@api.route('/<string:category>')
class Query(Resource):

    def get(self, category):
        return {'status': '구현 중'}


@api.route('/<string:answer_id>')  # question_id: Object_id
class Query(Resource):
    #  해당하는 답변과 매칭 된 질문들 보내기
    def get(self, answer_id):
        return {'status': '구현 중'}

    def patch(self, answer_id):
        return {'status': '구현 중'}

    def delete(self, answer_id):
        return {'status': '구현 중'}


@api.route('/rebase')
class Query(Resource):

    @api.doc('데이터베이스 리베이스')
    def get(self):
        return {'status': '구현 중'}
