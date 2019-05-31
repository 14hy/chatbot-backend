from bson.objectid import ObjectId
from flask_restplus import Resource, Namespace
from flask_restplus import reqparse
from src.services.search import Search

from src.main import Engine

engine = Engine()
_search = Search()
api = Namespace('v2/service/QA', description='Service/QA endpoint', ordered=True)


@api.route('/<string:text>')
class QA(Resource):

    @api.doc('주어진 텍스트와 데이터베이스를 사용하여 QA')
    def get(self, text):
        answer, output = _search.response(chat=text)
        return {'answer': answer,
                'context': output}


@api.route('/context')
class QA(Resource):

    @api.doc('주어진 질문과 문단을 사용하여 QA', params={'text': '질문', 'context': '문단'})
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('chat', required=True, type=str)
        parser.add_argument('context', required=True, type=str)
        args = parser.parse_args(strict=True)

        answer = _search.response_with_context(args['chat'], args['context'])
        return {'ansewr': answer,
                'context': args['context']}


@api.route('/_id')
class QA(Resource):

    @api.doc('주어진 질문과 문단 아이디를 사용하여 QA', params={'text': '질문', '_id': '문단아이디'})
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('chat', required=True, type=str)
        parser.add_argument('_id', required=True, type=str)
        args = parser.parse_args(strict=True)

        answer, context = _search.response_with_id(args['chat'], ObjectId(args['_id']))
        return {'answer': answer,
                'context': context}


@api.route('/subject')
class QA(Resource):

    @api.doc('주어진 질문과 문단 주제를 사용하여 QA', params={'text': '질문', 'subject': '문단주제'})
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text', required=True, type=str)
        parser.add_argument('subject', required=True, type=str)
        args = parser.parse_args(strict=True)
        answer, context = _search.response_with_subject(args['text'], args['subject'])

        return {'answer': answer,
                'context': context}
