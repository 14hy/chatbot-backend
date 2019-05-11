import os

BASE_DIR = '/home/rhodochrosited/'

PREPROCESS = {
    'vocab_file': os.path.join(BASE_DIR, 'squad_train_model/vocab-9171.txt'),
    'use_morphs': True,
    'max_seq_length-search': 384,
    'max_query_length-search': 64,
    'max_query_length-similarity': 25,  # = seq_length
    'keywords_tags': ['NNG', 'NNP', 'NNB', 'NNBC', 'MAG'],
    'clean_orig_tags': ['JK', 'JX', 'JC']
}

BERT = {
    'model_path-search': os.path.join(BASE_DIR, 'squad_train_model/model.ckpt-11000'),
    'model_path-similarity': os.path.join(BASE_DIR, 'data/model_1/model.ckpt-100000'),
    'bert_json': os.path.join(BASE_DIR, 'squad_train_model/bert_config.json'),
    'bert_json-ef': os.path.join(BASE_DIR, 'data/bert_config.json'),
    'similarity_layer': -2,
    'max_seq_length-search': 384,
    'max_seq_length-similarity': 25
}
TENSOR_SERVING = {
    'url-search': 'http://localhost:8501/v1/models/bert-search:predict',
    'url-similarity': 'http://localhost:8502/v1/models/similarity:predict'
}

HANDLER = {
    'search_threshold': 400  # search_threshold 이 값보다 클 시 search category로 넘어간다.
}

QUESTION = {
    'categories': ['shuttle_bus', 'food', 'talk', 'search', 'book']
}

SEARCH = {
    'sublinear_tf': True,
}

QUERY = {
    'distance': 'manhattan',
    'jaccard_threshold': 0.5
}

MONGODB = {
    'local_ip': 'localhost',
    'port': 27017,
    'db_name': 'chatbot',
    'col_questions': 'questions',
    'col_queries': 'queries',
    'col_contexts': 'contexts',
    'username': "mhlee",
    'password': "mhlee"
}

if __name__ == '__main__':
    print(BASE_DIR)
    print(PREPROCESS)
