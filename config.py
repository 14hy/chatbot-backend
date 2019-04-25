import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TEST3 = {
#     'vocab_file': os.path.join(BASE_DIR, '../data/vocab_mecab+khaiii_noBPE_5888'),
#     'max_seq_length': 25,
#     'max_query_length': 25,
#     'model_path': os.path.join(BASE_DIR, '../data/model_1/model.ckpt-100000'),
#     'bert_json': os.path.join(BASE_DIR, '../data/bert_config.json'),
#     'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
#     'feature_layers': -2,
#     'distance': 'manhattan' # euclidean, manhattan
# }
TEST = {
    'vocab_file': os.path.join(BASE_DIR, '../data/google_model/vocab.txt'),
    'max_seq_length': 25,
    'max_query_length': 25,
    'model_path': os.path.join(BASE_DIR, '../data/google_model/bert_model.ckpt'),
    'bert_json': os.path.join(BASE_DIR, '../data/google_model/bert_config.json'),
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

DEFAULT_CONFIG = TEST

MONGODB_CONFIG = {

    'local_ip': '127.0.0.1',
    'port': 27017,
    'db_name': 'chatbot',
    'col_questions': 'questions',
    'col_queries': 'queries'
}


if __name__ == '__main__':
    print(BASE_DIR)
