import pickle

from engine.db.index import *
from engine.db.queries.query import convert_to_query

_queries = db[MONGODB_CONFIG['col_queries']]


def insert(query):
    feature_vector = pickle.dumps(query.feature_vector)

    document = {
        'chat': query.chat,
        'feature_vector': feature_vector,
        'keywords': query.keywords,
        'matched_question': query.matched_question.text,
        # 저장은 object id로 하지만 query 객체는 question 객체 이므로 헷갈리지 말 것
        'manhattan_similarity': query.manhattan_similarity,
        'jaccard_score': query.jaccard_similarity
    }

    return _queries.update_one({'chat': document['chat']}, {'$set': document}, upsert=True)


def get_list():
    queries = []
    cursor = _queries.find({})

    for document in cursor:
        query = convert_to_query(document)
        queries.append(query)
    return queries
