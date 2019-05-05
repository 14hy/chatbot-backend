from flask_restplus import Resource
from flask_restplus import reqparse, fields, inputs
from flask import Response
from api.common.settings import *
from engine.db.mongo import PymongoWrapper
from engine.main import Engine

backend = Engine()
pw = PymongoWrapper()


@v1.route('/chat')
class CategorizeChat(Resource):

    @v1.doc(params={'chat': 'A chat'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('chat', type=str, required=True, help='사용자가 입력한 질문 및 발화')
            args = parser.parse_args()
            _chat = args['chat']
            _answer = backend.chat_to_answer(_chat)
            return _answer
        except Exception as err:
            {'status': 'error'}


@v1.route('/db/questions/add')
class Manager(Resource):

    @v1.doc('질문 추가', params={'text': '등록 할 질문', 'answer': '등록 할 답변(default=None)', 'category': '카테고리'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text', type=str, help='등록 할 질문')
            parser.add_argument('answer', type=str, required=False, help='답변(default=None)')
            parser.add_argument('category', type=str, required=True, help='카테고리')
            args = parser.parse_args(strict=True)

            _text = args['text']
            _answer = args['answer']
            _category = args['category']
            pw.create_question_and_insert(_text, _answer, _category)
            return {'status': 'Success'}
        except Exception as err:
            return {'status': 'error'}

@v1.route('/bus/shuttle')
class Shuttle(Resource):

    @v1.doc('셔틀 버스 정보 조회', params={'weekend': '휴일여부(True, False)', 'season': 'semester/ between/ vacation',
                                   'hours': '시간 - int(0~23)', 'minutes': '분 - int(0~59)', 'seconds': '초 - int(0~59)'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('weekend', required=True, type=inputs.boolean, help='휴일여부')
            parser.add_argument('season', required=True, type=str, help='학기/ 계절학기/ 방학')
            parser.add_argument('hours', type=int, required=True, help='시간(0~23)')
            parser.add_argument('minutes', type=int, required=True, help='분(0~59)')
            parser.add_argument('seconds', type=int, required=True, help='초(0~59)')
            args = parser.parse_args(strict=True)

            _weekend = args['weekend']
            _season = args['season']
            _hours = args['hours']
            _minutes = args['minutes']
            _seconds = args['seconds']

            return backend.get_shuttle(_weekend, _season, _hours, _minutes, _seconds)
        except Exception as err:
            print(err)
            return {'status': 'error'}

    def get(self):
        try:
            return backend.get_shuttle(current=True)
        except Exception as err:
            return {'status': 'error'}


# @v1.route('/db/questions/reset')
# class Manager(Resource):
#
#     @v1.doc('모든 질문 삭제')
#     def delete(self):
#         try:
#             pw.remove_all_questions()
#             return {'remove_all_questions': 'success'}
#         except Exception as err:
#             return {'error': err}


