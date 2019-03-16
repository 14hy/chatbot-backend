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
    'vocab_file': os.path.join(BASE_DIR, 'ckpt/fine_tuned/vocab.txt'),
    'max_seq_length': 384,
    'max_query_length': 64,
    'model_path': os.path.join(BASE_DIR, 'ckpt/pre_trained/bert_model.ckpt'),
    'bert_json': os.path.join(BASE_DIR, 'ckpt/pre_trained/bert_config.json'),
    'categories': ['셔틀', '밥', '잡담', '학사행정', '검색'],
    'feature_layers': -2
}

DEFAULT_CONFIG = TEST2

MONGODB_CONFIG = {
    'local_ip': '127.0.0.1',
    'port': 27017,
    'db_name': 'chatbot',
    'col_question': 'questions'
}

if __name__ == '__main__':
    print(BASE_DIR)
