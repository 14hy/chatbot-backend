from flask_restplus import Resource

from api.v2.preprocess import api
from src.data.preprocessor import PreProcessor

_preprocessor = PreProcessor()


@api.route('/clean/<string:text>')
class Clean(Resource):

    @api.doc('cleanning')
    def get(self, text):
        return {'cleaend': _preprocessor.clean(text)}
