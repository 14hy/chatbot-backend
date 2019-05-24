from flask_restplus import Resource

from api.v2.preprocess import api


@api.route('/keyword/<string:text>')
class Keyword(Resource):

    @api.doc('주어진 텍스트의 키워드 리스트')
    def get(self, text):
        return {'status': '구현 중'}
