from flask_restplus import Resource

from api.v2.visualize import api


@api.route('/line')
class Line(Resource):
    pass
