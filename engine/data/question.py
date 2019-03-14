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

        self.preprocessor = PreProcessor()

        self.bert_model = Model()
        self.bert_model.build_model()

    def create_question(self, text, category, answer=None):
        pass

if __name__ == '__main__':
    qm = QuestionMaker()