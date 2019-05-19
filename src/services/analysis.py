from collections import Counter

import config
import numpy as np
from src.data.query import *
from src.data.preprocessor import PreProcessor
from src.db.queries.index import get_list
from src.db.queries import index as _queries
from src.db.questions import index as _questions
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

_query_maker = QueryMaker()
_preprocessor = PreProcessor()

CONFIG = config.ANALYSIS


# plt.rcParams["font.family"] = 'NanumGothic'
# plt.rcParams["font.size"] = 5
# plt.rcParams['figure.figsize'] = (15, 15)


def get_Morphs(query):
    query, removed = _preprocessor.clean(query)
    output = _preprocessor.get_morphs(query)
    output['removed'] = removed
    return output


def get_JaccardSimilarity(query):
    query, removed = _preprocessor.clean(query)
    sorted_jaccard_list = _query_maker.get_jaccard(query)

    output = {
        'query': query,
        'removed': removed,
        'question_1': {
            'text': None,
            'only_in_query': None,
            'only_in_question': None,
            'in_both': None,
            'score': None
        },
        'question_2': {
            'text': None,
            'only_in_query': None,
            'only_in_question': None,
            'in_both': None,
            'score': None
        },
        'question_3': {
            'text': None,
            'only_in_query': None,
            'only_in_question': None,
            'in_both': None,
            'score': None
        }
    }
    N = 1
    _morphs_query = _preprocessor.get_morphs(query)
    _length_query = len(_morphs_query)
    for key, score in sorted_jaccard_list.items():
        _morphs_question = _preprocessor.get_morphs(key)
        only_in_query = {}
        only_in_question = {}
        in_both = {}
        for word, tag in _morphs_query.items():
            if word == 'text':
                continue
            if word in _morphs_question.keys():
                in_both[word] = tag
            else:
                only_in_query[word] = tag

        for word, tag in _morphs_question.items():
            if word == 'text':
                continue
            if word not in in_both.keys():
                only_in_question[word] = tag

        output['question_{}'.format(N)] = {
            'text': key,
            'only_in_query': only_in_query,
            'only_in_question': only_in_question,
            'in_both': in_both,
            'score': score
        }
        N += 1
        if N == 4:
            break
    return output


def get_MostCommonKeywords(n=7):
    # 자주 나오는 키워드 Top
    queries_list = get_list()

    keywords = []
    for query in queries_list:
        for keyword in query.keywords:
            keywords.append(keyword)
    most_common = Counter(keywords).most_common(n)

    output = {}

    for key, value in most_common:
        output[key] = value

    return output


def get_SearchToQuestion(n=20):
    queries = _queries.find_by_category('search')[:n]

    output = {}
    for query in queries:
        output[query.chat] = query.answer['answer']

    return output
    # search 중에서 정확도가 높은 것들을 사전 답변으로 옮기는 것을 고려


def visualize_similarity(chat, mode=0):
    """t-SNE 학습을 통해 벡터 시각화"""
    assert type(chat) == str
    tsne = TSNE(n_components=CONFIG['n_components'],
                perplexity=CONFIG['perplexity'],
                learning_rate=CONFIG['learning_rate'],
                n_iter=CONFIG['n_iter'],
                metric=CONFIG['metric'],
                method=CONFIG['method'])
    X = []  # (n_samples, n_features)
    X_text = []
    X_category = []

    chat_vector = _query_maker.modelWrapper.similarity(chat=chat)
    chat_vector = _query_maker.get_weighted_average_vector(text=chat, vector=chat_vector)

    questions_list = _questions.find_all()

    X.append(chat_vector)
    X_text.append(chat)
    X_category.append('입력')

    for question in questions_list:
        text = question.text
        question_vector = _query_maker.get_weighted_average_vector(text=text, vector=question.feature_vector)

        if type(question_vector) == np.ndarray:
            X.append(question_vector)
            X_text.append(text)
            X_category.append(question.category)

    Y = tsne.fit_transform(X=X)  # low-dimension vectors
    x = Y[:, 0]
    y = Y[:, 1]

    output = {}
    for category in CONFIG['categories']:
        temp = []
        for i in range(len(X_category)):
            if X_category[i] == category:
                temp.append({'text': X_text[i],
                             'x': x[i],
                             'y': y[i]})
        output[category] = temp
    # plt.scatter(x=x, y=y)
    # for i in range(len(x)):
    #     plt.text(x=x[i] + 0.1, y=y[i], s=X_text[i], fontsize=10)
    # plt.show()
    return output


if __name__ == '__main__':
    # print(get_JaccardSimilarity('셔틀 언제 오나요?'))
    # b = get_MostCommonKeywords()
    # print(get_SearchToQuestion())
    visualize_similarity('셔틀 언제 와?')
    pass
