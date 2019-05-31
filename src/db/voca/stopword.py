from src.db.index import *

collection = db[MONGODB_CONFIG['col_stopword']]


class StopWord(object):
    def __init__(self, word):
        self.word = word


def insert(word):
    doc = StopWord(word)
    return str(collection.update_one({'word': word}, update={'$set': doc.__dict__}, upsert=True))
