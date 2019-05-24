from flask_restplus import Resource
from api.v2.preprocess import api
from src.services.analysis import get_Morphs


@api.route('/tag/<string:text>')
class Tag(Resource):

    @api.doc('주어진 텍스트의 형태소 정보')
    def get(self, text):
        return get_Morphs(query=text)

