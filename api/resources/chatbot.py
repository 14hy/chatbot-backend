from flask_restplus import Resource
from flask_restplus import reqparse, fields
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
            parser.add_argument('chat', type=str, required=True, help='사용자가 입력한 질문및 발화')
            args = parser.parse_args()

            _chat = args['chat']
            _answer = backend.chat_to_answer(_chat)
            print('*** answer ***')
            print(_answer)
            return _answer
        except Exception as err:
            return {'error': err}


@v1.route('/db/add_question')
class Manager(Resource):

    @v1.doc('질문 추가', params={'text': '등록 할 질문', 'answer': '등록 할 답변(default=None)'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text', type=str, help='등록 할 질문')
            parser.add_argument('answer', type=str, required=False, help='답변, 비워도 됨')
            args = parser.parse_args(strict=True)

            _text = args['text']
            _answer = args['answer']
            pw.create_question_and_insert(_text, _answer)
            return {'created_question_and_insert': 'success'}
        except Exception as err:
            return {'error': err}


# 만들 api
'''
question list
question 생성, 삭제, 수정, 
이 질문이 답변 한 쿼리 리스트?

query list
query 삭제, 수정, 분석(?)


'''
