from flask_restplus import Resource

from api.v2.visualize import api
from src.services.analysis import get_FeatureSimilarity


@api.route('/table/cosine/<string:text>')
class Table(Resource):

    @api.doc('주어진 텍스트에 대한 코사인 유사도 순위')
    def get(self, text):
        return get_FeatureSimilarity(text)



