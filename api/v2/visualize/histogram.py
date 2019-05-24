from flask_restplus import Resource

from api.v2.visualize import api


@api.route('/histogram')
class Histogram(Resource):
    pass
