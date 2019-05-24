from flask_restplus import Resource, Namespace, inputs
from flask_restplus import reqparse

from src.main import Engine

engine = Engine()
api = Namespace('v2/service/bus', description='Service/bus endpoint')


@api.route('/')
class Bus(Resource):

    @api.doc('현재 시간을 기준으로 시간 조회')
    def get(self):
        return engine.get_shuttle(current=True)

    @api.doc('주어진 시간을 기준으로 시간 조회',
             params={'weekend': '휴일 여부 True, False', 'season': 'semester/ between/ vacation', 'hours': '시간 - (0~23)',
                     'minutes': '분 - (0~59)', 'seconds': '초 -(0~59)'})
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('weekend', required=True, type=inputs.boolean, help='휴일여부')
        parser.add_argument('season', required=True, type=str, help='학기/ 계절학기/ 방학')
        parser.add_argument('hours', type=int, required=True, help='시간(0~23)')
        parser.add_argument('minutes', type=int, required=True, help='분(0~59)')
        parser.add_argument('seconds', type=int, required=True, help='초(0~59)')
        args = parser.parse_args(strict=True)

        _weekend = args['weekend']
        _season = args['season']
        _hours = args['hours']
        _minutes = args['minutes']
        _seconds = args['seconds']
        return engine.get_shuttle(_weekend, _season, _hours, _minutes, _seconds)
