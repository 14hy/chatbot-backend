from flask_restplus import Resource
from api.v2.preprocess import api


@api.route('/token<string:text>')
class Token(Resource):

    @api.doc('주어진 텍스트의 토큰화')
    def get(self, text):
        return {'status': '구현 중'}


