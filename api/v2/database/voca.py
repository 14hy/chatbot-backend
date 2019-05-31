from flask_restplus import Resource, Namespace, reqparse
from src.db.voca import julimmal as _julimmal
from src.db.voca import stopword as _stopword
from .utils import *
from bson import ObjectId

api = Namespace('v2/db/voca', 'DataBase Query endpoint')


@api.route('/stopword')
class Voca(Resource):

    @api.doc('불용어 목록 불러오기')
    def get(self):
        return cursor_to_json(_stopword.collection.find({}))


@api.route('/stopword/<string:word>')
class Voca(Resource):

    def post(self, word):
        return {'status': _stopword.insert(word)}


@api.route('/stopword/<string:_id>')
class Voca(Resource):

    def delete(self, _id):
        return {'status': _stopword.collection.delete_one({'_id': ObjectId(_id)})}


@api.route('/jul-immal')
class Voca(Resource):

    @api.doc('줄임말 목록 불러오기')
    def get(self):
        return cursor_to_json(_julimmal.collection.find({}))

    @api.doc('줄임말 추가하기', params={'orig': '원래 단어', 'sub': '교체 될 단어'})
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('orig', type=str, required=True, help='원래 단어')
        parser.add_argument('sub', type=str, required=True, help='교체 될 단어')
        args = parser.parse_args(strict=True)

        orig = args['orig']
        sub = args['sub']

        return {'status': _julimmal.insert(orig, sub)}

@api.route('/jul-immal/<string:_id>')
class Voca(Resource):

    @api.doc('줄임말 제거')
    def delete(self, _id):
        return {'status': _julimmal.collection.delete_one({'_id': ObjectId(_id)})}

