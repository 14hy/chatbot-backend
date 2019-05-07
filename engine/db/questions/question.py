class Question(object):
    def __init__(self, text, category, answer, feature_vector,
                 keywords, morphs, object_id=None):
        '''

        :param text: str, 질문
        :param category: str, config에서 정의된 카테고리 이여야 함
        :param answer: str/ None, 없어도 됨
        :param feature_vector: [784] Feature vector
        '''
        self.keywords = keywords
        self.object_id = object_id
        self.category = category
        self.answer = answer
        self.text = text
        self.feature_vector = feature_vector
        self.morphs = morphs

