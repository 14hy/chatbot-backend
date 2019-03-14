# 1. 질문이 들어오면 저장
# 2. 질문 전처리 및 특징 추출
# 3. 질문 분류 및

class Query(object):
    def __init__(self, chat, category=None, response=None):

        self.response = response
        self.chat = chat
        self.category = category

class Categorizer():
    '''

    '''
    def __init__(self):
        pass

    def categorize(self, query):
        '''
        유사도분석을 통해서 카테고리화 한다.
        :param query:
        :return:

        사전에 준비 된 질문과의 유사도를 비교
        비효율적이고 계산이 비싸기 때문에 충분한 데이터가 쌓이면
        classification model을 만들어서 대체하기
        혹은 사용자에게 1차 분류를 시킴으로써 계산량 감소 가능
        '''




        # 1. 코사인 거리
        # 2. 맨하탄 거리
        # 3. 유클리드 거리
        # 4. 자카드 유사도


        # query의 챗을 피쳐로,

        # 추출한 피쳐를 바탕으로 카테고리 추출
        # ㄴ 다양한 유사도 비교 방법 Feature extractor(?)
        # 피쳐 벡터, 명사리스트, 동사리스트?'
        #
        #

        return query

class ChatHandler():
    '''
    preprocess chat to query
    '''
    def __init__(self):
        self.categorizer = Categorizer()
        pass

    def create_query_from_chat(self, chat):
        '''
        :param chat: str
        :return: Query object
        '''
        query = Query(chat)

        return self.categorizer.categorize(query)
        # chat to feature vector

        # categorize

        #
        pass