from flask_restplus import Resource, Namespace
from flask_restplus import reqparse
from src.db.contexts import index as _context
from .utils import id_to_str

api = Namespace(name='v2/db/context', description='DataBase Context endpoint')


@api.route('/')
class Context(Resource):

    @api.doc('문단들의 리스트')
    def get(self):
        return id_to_str(list(_context.collection.find({})))

    @api.doc('문단 추가', params={'subject': '문단의 주제, 중복X', 'text': '문단'})
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('subject', type=str, help='등록 할 문단의 주제')
        parser.add_argument('text', type=str, required=True, help='등록 할 문단')
        args = parser.parse_args(strict=True)

        return {'status': str(_context.create_insert(text=args['text'], subject=args['subject']))}


@api.route('/<string:_id>')
class Context(Resource):

    def get(self, _id):
        return id_to_str(_context.collection.find_one({'_id': _id}))

    def delete(self, _id):
        return {'status': str(_context.collection.delete_one({'_id': _id}))}

    @api.doc('문단 수정', params={'subject': '문단의 주제, 중복X', 'text': '문단'})
    def patch(self, _id):
        parser = reqparse.RequestParser()
        parser.add_argument('subject', type=str, required=False, default=None, help='수정된 문단의 주제')
        parser.add_argument('text', type=str, required=False, default=None, help='수정된 문단')
        args = parser.parse_args(strict=True)

        context = args['context']
        subject = args['subject']

        target = _context.collection.find_one({'_id': _id})

        if subject:
            target['subject'] = subject
        if context:
            target['context'] = context

        return {'status': str(_context.collection.update_one({'_id': _id}, update=target))}


@api.route('/<string:subject>')
class Context(Resource):
    pass

    def get(self, subject):
        return _context.collection.find_one({'subject': subject})

    def delete(self, subject):
        return {'status': str(_context.collection.delete_one({'subject': subject}))}

    @api.doc('문단 수정', params={'subject': '문단의 주제, 중복X', 'text': '문단'})
    def patch(self, subject):
        parser = reqparse.RequestParser()
        parser.add_argument('subject', type=str, required=False, default=None, help='수정된 문단의 주제')
        parser.add_argument('text', type=str, required=False, default=None, help='수정된 문단')
        args = parser.parse_args(strict=True)

        context = args['context']
        subject = args['subject']

        target = _context.collection.find_one({'subject': subject})

        if subject:
            target['subject'] = subject
        if context:
            target['context'] = context

        return {'status': str(_context.collection.update_one({'subject': subject}, update=target))}


@api.route('/<string:text>')
class Context(Resource):
    pass
