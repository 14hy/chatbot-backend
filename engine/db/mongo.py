from pymongo import MongoClient
from configparser import ConfigParser


class PymongoWrapper():
    def __init__(self):

        config = ConfigParser()
        config.read('../../conf/config.ini')
        self.db_config = {}
        self.db_config["local_ip"] = config['MONGODB']['local_ip']
        self.db_config["port"] = config['MONGODB']['port']

        self._client = MongoClient(self.db_config['local_ip'], self.db_config['port'])

    def insert_question(self, question):
        '''

        :param question: Question object
        :return:
        '''
        if question.feature_vector is None:
            raise Exception('질문의 feature vector None')

        document = {'text': }

        pass


if __name__ == '__main__':
    pw = PymongoWrapper()
