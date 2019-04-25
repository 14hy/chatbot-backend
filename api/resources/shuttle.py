from flask_restplus import Resource
from flask_restplus import reqparse

from api.common.settings import *


@v1.route('/shuttle')
class ShuttleBus(Resource):

    @v1.doc(params={'chat': 'A chat'})
    @v1.marshal_list_with(answer)
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('chat', type=str, required=True, help='사용자가 입력한 질문및 발화')
            args = parser.parse_args()

            _chat = args['chat']
            _answer = backend.chat_to_answer(_chat)
            return {'answer': 'hello?'}
        except Exception as err:
            return {'error': err}
