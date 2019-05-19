from src.db.contexts.context import *
from src.db.index import *

_contexts = db[MONGODB_CONFIG['col_contexts']]


def create_insert(text, subject):
    documnet = convert_to_document(text, subject)
    return insert(documnet)


def insert(document):
    return _contexts.update_one({'subject': document['subject']},
                                {'$set': document}, upsert=True)


def find_all():
    contexts = _contexts.find({})
    return contexts


def find_by_subject(_subject):
    document = _contexts.find_one({'subject': _subject})
    return document
