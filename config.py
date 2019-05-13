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
    'model_path-similarity': os.path.join(BASE_DIR, 'data/pretrain_512_3/model.ckpt-550000'),
    'bert_json': os.path.join(BASE_DIR, 'squad_train_model/bert_config.json'),
    'similarity_layer': -2,
    'version-similarity': 1,
    'version-search': 1,
    'max_seq_length-search': 384,
    'max_seq_length-similarity': 25,
    'MODEL_DIR': os.path.join(BASE_DIR, 'hdd2/tensor_serving_models')
}
TENSOR_SERVING = {
    'url-search': 'http://localhost:8501/v1/models/search:predict',
    'url-similarity': 'http://localhost:8502/v1/models/similarity:predict'
}

HANDLER = {
    'DUMMY': None
}

QUESTION = {
    'categories': ['shuttle_bus', 'food', 'talk', 'search', 'book', 'prepared']
}

SEARCH = {
    'sublinear_tf': True,
}

QUERY = {
    'distance': 'manhattan',
    'jaccard_threshold': 0.5,
    'search_threshold': 120
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
