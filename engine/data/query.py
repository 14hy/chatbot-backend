

class Query(object): # TODO 어떤 정보가 필요 할 지 계속 고민 해보기
    def __init__(self, chat, feature_vector, keywords, matched_question, feature_distance, jaccard_score):
        '''

        :param chat: str
        :param feature_vector: array
        :param keywords: list
        :param matched_question: question object
        :param feature_distance: float
        '''
        self.chat = chat
        self.feature_vector = feature_vector
        self.keywords = keywords
        self.matched_question = matched_question  # 어떤 질문과 매칭 되었었는지.
        self.feature_distance = feature_distance  # 거리는 어떠 하였는 지
        self.jaccard_distance = jaccard_score

    def __str__(self):
        return 'Query chat:%5s, keywords%10s, question:%5s, feature_distance:%5.3f, jaccard_distance:%5.3f' \
               % (self.chat, self.keywords, self.matched_question.text, self.feature_distance, self.jaccard_distance)

