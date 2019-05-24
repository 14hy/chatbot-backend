from flask_restplus import Resource, Namespace
from flask_restplus import reqparse
from src.main import Engine
from bson.objectid import ObjectId
import config
from src.data.question import QuestionMaker

_question_maker = QuestionMaker()
CONFIG = config.QUESTION
engine = Engine()
QUESTIONS = 0
QUERIES = 1

api = Namespace('v2/db/answer', 'DataBase Answer endpoint')


@api.route('/')
class Questions(Resource):

    @api.doc('모든 답변들의 id, 질문')
    def get(self):
        return {'status': '구현 중'}

    @api.doc('답변 추가/ 수정', params={'text': '등록 할 답변의 질문', 'answer': '답변', 'category': str(CONFIG['categories'])})
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text', type=str, required=True)
        parser.add_argument('answer', type=str, required=False, help='답변(default=None)')
        parser.add_argument('category', type=str, required=True, choices=CONFIG['categories'], help='카테고리')
        args = parser.parse_args(strict=True)

        engine.insert_question(args['text'], args['answer'], args['category'])
        return {'status': 'success'}


@api.route('/<string:text>')
class Questions(Resource):

    @api.doc('텍스트가 포함 된 답변리스트')
    def get(self, text):
        return {'status': '구현 중'}


@api.route('/<string:_id>')
class Questions(Resource):

    @api.doc('해당하는 아이디의 답변', params={'_id': 'String'})
    def get(self, _id):
        _id = ObjectId(_id)
        return {'status': '구현 중'}

    @api.doc('답변의 질문텍스트, 답변, 카테고리 수정')
    def patch(self, _id):
        _id = ObjectId(_id)
        return {'status': '구현 중'}

    @api.doc('해당 아이디 답변 삭제')
    def delete(self, _id):
        _id = ObjectId(_id)
        return {'status': '구현 중'}


@api.route('/rebase')
class Questions(Resource):

    @api.doc('데이터베이스 리베이스')
    def get(self):
        _question_maker.rebase()
        return {'status': 'done'}
