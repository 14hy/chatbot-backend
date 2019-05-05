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
            parser.add_argument('chat', type=str, required=True, help='사용자가 입력한 질문 및 발화')
            args = parser.parse_args()
            _chat = args['chat']
            _answer = backend.chat_to_answer(_chat)
            return _answer
        except Exception as err:
            pass


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
            return {'err': err}

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

# 만들 api
'''
question list
question 생성, 삭제, 수정, 
이 질문이 답변 한 쿼리 리스트?

query list
query 삭제, 수정, 분석(?)


'''
