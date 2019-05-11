from engine.data.query import QueryMaker
from engine.db.index import *
from engine.db.queries.query import *

_queries = db[MONGODB_CONFIG['col_queries']]
_query_maker = QueryMaker()


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


def rebase():
    for document in _queries.find({}):
        _id = document['_id']
        chat = document['chat']
        try:
            added_time = document['added_time']
        except KeyError:
            added_time = None

        try:

            query = _query_maker.make_query(chat=chat,
                                            added_time=None)
            insert(query)
            _queries.delete_one({'_id': _id})
            print('rebase: {}'.format(query.chat))
        except Exception as err:
            print('rebase ERROR: ', err)
            print(document)
            return document


if __name__ == '__main__':
    rebase()
