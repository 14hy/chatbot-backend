import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST1 = {
    'vocab_file': os.path.join(BASE_DIR, 'ckpt/fine_tuned/vocab.txt'),
    'max_seq_length': 384,
    'max_query_length': 64,
    'bert_json': os.path.join(BASE_DIR, 'ckpt/fine_tuned/bert_config.json'),
    'model_path': os.path.join(BASE_DIR, 'ckpt/fine_tuned/model.ckpt-30203'),
    'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
    'feature_layers': -2
}

TEST2 = {
    'vocab_file': os.path.join(BASE_DIR, 'ckpt/pre_trained/vocab.txt'),
    'max_seq_length': 384,
    'max_query_length': 64,
    'model_path': os.path.join(BASE_DIR, 'ckpt/pre_trained/bert_model.ckpt'),
    'bert_json': os.path.join(BASE_DIR, 'ckpt/pre_trained/bert_config.json'),
    'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
    'feature_layers': -1
}

TEST3 = {
    'vocab_file': os.path.join(BASE_DIR, '../data/vocab_mecab+khaiii_noBPE_5888'),
    'max_seq_length': 25,
    'max_query_length': 25,
    'model_path': os.path.join(BASE_DIR, '../data/model_1/model.ckpt-100000'),
    'bert_json': os.path.join(BASE_DIR, '../data/bert_config.json'),
    'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
    'feature_layers': -2,
    'distance': 'manhattan' # euclidean, manhattan
}

# TEST4 = {
#     'vocab_file': os.path.join(BASE_DIR, '../data/vocab_mecab+khaiii_noBPE_5888'),
#     'max_seq_length': 128,
#     'max_query_length': 128,
#     'model_path': os.path.join(BASE_DIR, '../data/model_2/model.ckpt-200000'),
#     'bert_json': os.path.join(BASE_DIR, '../data/bert_config.json'),
#     'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
#     'feature_layers': -1,
#     'distance': 'manhattan' # euclidean, manhattan
# }

DEFAULT_CONFIG = TEST3

MONGODB_CONFIG = {

    'local_ip': '127.0.0.1',
    'port': 27017,
    'db_name': 'chatbot',
    'col_questions': 'questions',
    'col_queries': 'queries'
}


if __name__ == '__main__':
    print(BASE_DIR)
