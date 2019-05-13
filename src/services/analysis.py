from collections import Counter

from src.data.query import QueryMaker
from src.data.preprocess import PreProcessor
from src.db.queries.index import get_list
from src.db.queries import index as _queries

_query_maker = QueryMaker()
_preprocessor = PreProcessor()


def get_JaccardSimilarity(query):
    sorted_jaccard_list = _query_maker.get_jaccard(query)

    output = {
        'query': query,
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
            if word == 'output':
                continue
            if word in _morphs_question.keys():
                in_both[word] = tag
            else:
                only_in_query[word] = tag

        for word, tag in _morphs_question.items():
            if word == 'output':
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


def get_SearchToQuestion():
    queries = _queries.find_by_category('search')

    output = {}
    for query in queries:
        output[query.chat] = query.answer['answer']

    return output
    # search 중에서 정확도가 높은 것들을 사전 답변으로 옮기는 것을 고려


if __name__ == '__main__':
    print(get_JaccardSimilarity('셔틀 언제 오나요?'))
    b = get_MostCommonKeywords()
    print(get_SearchToQuestion())
