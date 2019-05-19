import os

BASE_DIR = '/home/rhodochrosited/'

PREPROCESS = {
    'vocab_file': os.path.join(BASE_DIR, 'squad_train_model/vocab-9171.txt'),
    'stop_words_file': os.path.join(BASE_DIR, 'chatbot', 'stop_words.txt'),
    'use_morphs': True,
    'max_seq_length-search': 384,
    'max_query_length-search': 64,
    'max_query_length-similarity': 25,  # = seq_length
    'keywords_tags': ['NNG', 'NNP', 'NNB', 'NNBC', 'MAG'],
    'clean_tags': ['JK', 'JX', 'JC'],
    'sub_file': os.path.join(BASE_DIR, 'chatbot', 'sub.txt')
}

BERT = {  # 새로운 TENSOR SERVING 모델을 만들 때 사용
    'model_path-search': os.path.join(BASE_DIR, 'hdd2/FINAL_SQUAD/model.ckpt-12408'),
    'model_path-similarity': os.path.join(BASE_DIR, 'hdd2/FINAL_PRETRAIN/model.ckpt-990000'),
    'bert_json': os.path.join(BASE_DIR, 'squad_train_model/bert_config.json'),
    'similarity_layer': -1,
    # ELMO LIKE FEATURE VECTOR LAYERS 여러레이어를 더하거나, -2, -3... 하위 레이어 만을 사용 해보는 방법들 시도해보기
    'version-similarity': 2,
    # 1: 128 seq length 75000 step
    # 2: 512 seq length 990000 step
    'version-search': 2,
    # 1: F1 score 71
    # 2: F1 score 83.6 + train+dev -> (92)
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
    'max_context_num': 3  # search에서 tf idf 로 최대한 찾을 문단의 수
}

ANALYSIS = {
    # T-SNE
    'perplexity': 30.0,  # 5~50 권장
    'learning_rate': 200.0,  # 10~1000
    'n_iter': 1000,  # 최소 250
    'metric': 'euclidean',  # distance metric ( x_i <-> x_j )
    'method': 'barnes_hut',  # 속도개선
    'n_components': 2,  # y 차원
}

QUERY = {
    'distance': 'euclidean',
    'jaccard_threshold': 0.55,
    'search_threshold': 15,
    'idf_weight': 0.1,
    # idf_weight: 건드리면 search threshold 도 조정 해줘야 하며
    # 높일 수록 더욱 더 비슷한 것들만 찾게 됨
    'metric': ANALYSIS['metric']
}
MONGODB = {
    'ip': 'localhost',
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
