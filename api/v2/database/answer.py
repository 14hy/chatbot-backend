from flask_restplus import Resource, Namespace
from flask_restplus import reqparse
from src.main import Engine
import config
from src.data.question import QuestionMaker
from src.db.questions import index as _questions
from .utils import *

_question_maker = QuestionMaker()
CONFIG = config.QUESTION
engine = Engine()
QUESTIONS = 0
QUERIES = 1

api = Namespace('v2/db/answer', 'DataBase Answer endpoint')

filter = {'_id': 1, 'text': 1, 'answer': 1}


@api.route('/')
class Questions(Resource):

    @api.doc('모든 답변들의 id, 질문')
    def get(self):
        return id_to_str(list(_questions.collection.find({}, filter)))

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
        return id_to_str(
            list(_questions.collection.find({'$text': {'$search': text}}, filter)))


@api.route('/<string:_id>')
class Questions(Resource):

    @api.doc('해당하는 아이디의 답변', params={'_id': 'String'})
    def get(self, _id):
        return _questions.collection.find_one({'_id': _id})

    @api.doc('답변의 질문텍스트, 답변, 카테고리 수정', params={'text': '질문', 'answer': '답변', 'category': '카테고리'})
    def patch(self, _id):
        parser = reqparse.RequestParser()
        parser.add_argument('text', required=False, default=None, help='수정 될 질문')
        parser.add_argument('answer', required=False, default=None, help='수정 될 답변')
        parser.add_argument('category', required=False, default=None, help='수정 될 카테고리')
        args = parser.parse_args(strict=True)

        text = args['text']
        answer = args['answer']
        category = args['category']

        target = _questions.collection.find_one({'_id': _id})
        if text:
            target['text'] = text
        if answer:
            target['answer'] = answer
        if category:
            target['category'] = category

        return {'status': _questions.collection.update_one({'_id': _id}, update=target)}

    @api.doc('해당 아이디 답변 삭제')
    def delete(self, _id):
        return {'status': '구현 중'}


@api.route('/rebase')
class Questions(Resource):

    @api.doc('데이터베이스 리베이스')
    def get(self):
        _question_maker.rebase()
        return {'status': 'done'}
