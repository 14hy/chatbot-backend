from flask_restplus import Resource

from api.v2.visualize import api
from src.services.analysis import get_JaccardSimilarity


@api.route('/diagram/tag/<string:text>')
class Diagram(Resource):

    @api.doc('주어진 텍스트의 자카드 유사도')
    def get(self, text):
        return get_JaccardSimilarity(query=text)
