def cursor_to_json(cursor):
    cursor = list(cursor)
    for document in cursor:
        document['_id'] = str(document['_id'])
    return cursor


def doc_to_json(doc):
    doc['_id'] = str(doc['_id'])
    return doc
