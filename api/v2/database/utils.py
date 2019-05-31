def id_to_str(documents):
    for document in documents:
        document['_id'] = str(document['_id'])
    return documents
