import config
from src.data.preprocess import PreProcessor
from src.db.questions.question import Question
from src.model.serving import TensorServer


class QuestionMaker(object):

    def __init__(self):
        self.CONFIG = config.QUESTION
        self.model_wrapper = TensorServer()
        self.preprocessor = PreProcessor()

    def create_question(self, text, category=None, answer=None):
        if category == 'Circle' or category == 'noanswer':
            category = 'prepared'
        if category not in self.CONFIG['categories']:
            raise Exception('category must be ', self.CONFIG['categories'])

        keywords = self.preprocessor.get_keywords(text)
        morphs = self.preprocessor.get_morphs(text)
        vector = self.model_wrapper.similarity(text)

        return Question(text, category, answer, vector, morphs, keywords=keywords)


if __name__ == '__main__':
    qm = QuestionMaker()
