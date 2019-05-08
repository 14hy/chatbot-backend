from pprint import pprint
from time import time

from engine.data.handler import Handler
from engine.model.bert import Model
from engine.services.shuttle import ShuttleBus


class Engine(object):

    def __init__(self):

        self.chat_handler = Handler()
        self.shuttleBus = ShuttleBus()

    def get_shuttle(self, weekend=None, season=None, hours=None, minutes=None, seconds=None, current=False):
        if current:
            return self.shuttleBus.response()
        else:
            return self.shuttleBus.custom_response(weekend, season, hours, minutes, seconds)

    def chat_to_answer(self, chat):
        '''
        프론트로부터 질문을 받아 적절한 답변을 보냄
        :param chat: str
        :return: str
        '''
        tic = time()

        # TODO Query Feature extractor.
        answer = self.chat_handler.get_answer(chat)
        toc = time()
        # pprint(answer)

        return answer


if __name__ == '__main__':
    main = Engine()
    print('ENGINE MAIN')
