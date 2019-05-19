from time import time

from src.data.handler import Handler
from src.data.question import QuestionMaker
from src.services.shuttle import ShuttleBus
from src.data.query import QueryMaker
from src.services.search import *


class Engine(object):

    def __init__(self):

        self._chat_handler = Handler()
        self._shuttleBus = ShuttleBus()
        self._query_maker = QueryMaker()
        self._question_maker = QuestionMaker()
        self._search = Search()

    def get_shuttle(self, weekend=None, season=None, hours=None, minutes=None, seconds=None, current=False):
        if current:
            return self._shuttleBus.response()
        else:
            return self._shuttleBus.custom_response(weekend, season, hours, minutes, seconds)

    def chat_to_answer(self, chat):
        '''
        프론트로부터 질문을 받아 적절한 답변을 보냄
        :param chat: str
        :return: str
        '''
        tic = time()

        # TODO Query Feature extractor.
        answer = self._chat_handler.handle(chat)
        toc = time()
        # pprint(answer)

        return answer

    def insert_question(self, _text, _answer, _category):
        self._question_maker.insert_text(_text, _answer, _category)

    def chat_search(self, _chat, _subject):
        answer, context = self._search.response_with_context(_chat=_chat, _subject=_subject)
        return {
            'mode': 'chat_search',
            'answer': answer,
            'context': context
        }


if __name__ == '__main__':
    main = Engine()
