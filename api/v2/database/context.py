from flask_restplus import Resource, Namespace
from flask_restplus import reqparse
from src.db.contexts import index as _context

api = Namespace(name='v2/db/context', description='DataBase Context endpoint')


@api.route('/')
class Context(Resource):

    @api.doc('문단들의 리스트')
    def get(self):
        return {'status': '구현 중'}

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('subject', type=str, help='등록 할 문단의 주제')
        parser.add_argument('text', type=str, required=True, help='등록 할 문단')
        args = parser.parse_args(strict=True)

        _context.create_insert(text=args['text'], subject=args['subject'])
        return {'status': 'success'}
        #  body: 주제, 문단

    def patch(self):
        return {'status': '구현 중'}


@api.route('/<string:_id>')
class Context(Resource):
    pass

    def get(self, _id):
        return {'status': '구현 중'}

    def delete(self, _id):
        return {'status': '구현 중'}

    def patch(self, _id):
        return {'status': '구현 중'}


@api.route('/<string:subject>')
class Context(Resource):
    pass

    def get(self, subject):
        return {'status': '구현 중'}

    def delete(self, subject):
        return {'status': '구현 중'}

    def patch(self, subject):
        return {'status': '구현 중'}


@api.route('/<string:text>')
class Context(Resource):
    pass
