import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESS = {
    'vocab_file': os.path.join(BASE_DIR, '../squad_train_model/vocab-9171.txt'),
    'use_morphs': True,
    'max_seq_length': 384,
    'max_query_length': 64,
    'keywords_tags': ['NNG', 'NNP', 'NNB', 'NNBC'],
    'clean_orig_tags': ['JK', 'JX', 'JC']
}

BERT = {
    'model_path': os.path.join(BASE_DIR, '../squad_train_model/model.ckpt-11000'),
    'bert_json': os.path.join(BASE_DIR, '../squad_train_model/bert_config.json'),
    'feature_layers': -2,
    'max_seq_length': 384,
    'max_query_length': 64,
}
# BERT = {
#     'model_path': os.path.join(BASE_DIR, '../data/model_1/model.ckpt-100000'),
#     'bert_json': os.path.join(BASE_DIR, './ckpt/bert_config.json'),
#     'feature_layers': -2,
#     'max_seq_length': 25,
#     'max_query_length': 25,
# }
# PREPROCESS = {
#     'vocab_file': os.path.join(BASE_DIR, './ckpt/vocab_mecab+khaiii_noBPE_5888'),
#     'use_morphs': True,
#     'max_seq_length': 25,
#     'max_query_length': 25,
#     'keywords_tag': ['NNG', 'NNP', 'NNB', 'NNBC'],
# }

HANDLER = {
    'search_threshold': 400 # search_threshold 이 값보다 클 시 search category로 넘어간다.
}

QUESTION = {
    'categories': ['shuttle_bus', 'food', 'talk', 'search', 'book']
}

SEARCH = {
    'sublinear_tf': True,
}

QUERY = {
    'distance': 'manhattan',
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
