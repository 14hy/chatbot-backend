from src.db.contexts.context import *
from src.db.index import *

collection = db[MONGODB_CONFIG['col_contexts']]


def create_insert(text, subject):
    documnet = convert_to_document(text, subject)
    return insert(documnet)


def insert(document):
    return collection.update_one({'subject': document['subject']},
                                 {'$set': document}, upsert=True)


def find_all():
    contexts = collection.find({})
    return contexts


def find_by_subject(_subject):
    document = collection.find_one({'subject': _subject})
    return document
