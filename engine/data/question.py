import config
from engine.data.preprocess import PreProcessor
from engine.model import Model

class Question(object):
    def __init__(self, text, category, answer, feature_vector):
        self.category = category
        self.answer = answer
        self.text = text
        self.feature_vector = feature_vector

class QuestionMaker(object):
    def __init__(self):

        self.DEFAULT_CONFIG = config.DEFAULT_CONFIG

        self.preprocessor = PreProcessor()

        self.bert_model = Model()
        self.bert_model.build_model()

    def create_question(self, text, category, answer=None):
        '''

        :param text: 질문
        :param category: '셔틀', '학식', '잡담', '학사행정', '검색' ...
        :param answer: 정답/ None
        :return: Question object
        '''
        categories = self.DEFAULT_CONFIG['categories']

        if category not in categories:
            raise Exception('category must be ', categories)

        input_feature = self.preprocessor.create_InputFeature(text)
        feature_vector = self.bert_model.extract_feature_vector(input_feature)

        return Question(text, category, answer, feature_vector)

if __name__ == '__main__':
    qm = QuestionMaker()