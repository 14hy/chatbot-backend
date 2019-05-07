import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESS = {
    'vocab_file': os.path.join(BASE_DIR, '../squad_train_model/vocab-9171.txt'),
    'use_morphs': True,
    'max_seq_length': 25,
    'max_query_length': 25,
}

BERT = {
    'model_path': os.path.join(BASE_DIR, '../squad_train_model/model.ckpt-11000'),
    'bert_json': os.path.join(BASE_DIR, '../squad_train_model/bert_config.json'),
    'feature_layers': -2,
    'max_seq_length': 25,
    'max_query_length': 25,
}

HANDLER = {
    'distance': 'manhattan'  # euclidean, manhattan
}

QUESTION = {
    'categories': ['shuttle_bus', 'food', 'talk', 'search', 'book']
}

MONGODB = {
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
