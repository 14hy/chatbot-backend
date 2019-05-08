from engine.db.index import *
from engine.db.queries.query import *

_queries = db[MONGODB_CONFIG['col_queries']]


def insert(query):
    query = convert_to_document(query=query)
    return _queries.update_one({'chat': query['chat']}, {'$set': query}, upsert=True)


def get_list():
    queries = []
    cursor = _queries.find({})

    for document in cursor:
        query = convert_to_query(document)
        queries.append(query)
    return queries
