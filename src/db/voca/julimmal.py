from src.db.index import *

collection = db[MONGODB_CONFIG['col_julimmal']]


class JULIMMAL(object):

    def __init__(self, orig, sub):
        self.orig = orig
        self.sub = sub


def insert(orig, sub):
    doc = JULIMMAL(orig, sub)
    return str(collection.update_one({'orig': orig}, update={'$set': doc.__dict__}, upsert=True))
