from flask_restplus import Resource

from api.v2.preprocess import api


@api.route('/clean/<string:text>')
class Clean(Resource):

    @api.doc('cleanning')
    def get(self, text):
        return {'status': '구현 중'}
