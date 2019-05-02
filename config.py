import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST = {
    'vocab_file': os.path.join(BASE_DIR, './ckpt/vocab_mecab+khaiii_noBPE_5888'),
    'max_seq_length': 25,
    'max_query_length': 25,
    'model_path': os.path.join(BASE_DIR, '../data/model_1/model.ckpt-100000'),
    'bert_json': os.path.join(BASE_DIR, './ckpt/bert_config.json'),
    'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
    'feature_layers': -2,
    'distance': 'manhattan', # euclidean, manhattan
    'use_morphs': True
}

DEFAULT_CONFIG = TEST

MONGODB_CONFIG = {

    'local_ip': 'localhost',
    'port': 27017,
    'db_name': 'chatbot',
    'col_questions': 'questions',
    'col_queries': 'queries',
    'username': "mhlee",
    'password': "mhlee"
}


if __name__ == '__main__':
    print(BASE_DIR)
