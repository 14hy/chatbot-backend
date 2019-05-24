from flask_restplus import Resource, Namespace
from api.common.settings import *
from src.main import Engine

engine = Engine()

api = Namespace(name='v2', description='version-2')


@api.route('/<string:chat>')
class Index(Resource):

    @api.doc('챗봇 입력', params={'chat': '질문'})
    def get(self, chat):
        return engine.chat_to_answer(chat)

