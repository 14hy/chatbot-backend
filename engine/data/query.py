

class Query(object): # TODO 어떤 정보가 필요 할 지 계속 고민 해보기
    def __init__(self, chat, feature_vector, keywords, matched_question, manhattan_similarity, jaccard_similarity):
        '''

        :param chat: str
        :param feature_vector: array
        :param keywords: list
        :param matched_question: question object
        :param manhattan_similarity: None
        '''
        self.chat = chat
        self.feature_vector = feature_vector
        self.keywords = keywords
        self.matched_question = matched_question  # 어떤 질문과 매칭 되었었는지.
        self.manhattan_similarity = manhattan_similarity  # 거리는 어떠 하였는 지
        self.jaccard_similarity = jaccard_similarity

    def __str__(self):
        return 'Query chat:%5s, keywords%10s, question:%5s, feature_distance:%5.3f, jaccard_distance:%5.3f' \
               % (self.chat, self.keywords, self.matched_question.text, self.manhattan_similarity, self.jaccard_distance)

