import config
from engine.data.preprocess import PreProcessor
from engine.db.questions.question import Question
from engine.model.serving import TensorServer


class QuestionMaker(object):

    def __init__(self):
        self.CONFIG = config.QUESTION
        self.model_wrapper = TensorServer()
        self.preprocessor = PreProcessor()

    def create_question(self, text, category=None, answer=None):
        '''

        :param text: 질문
        :param category: '셔틀', '학식', '잡담', '학사행정', '검색' ...
        :param answer: 정답/ None
        :return: Question object
        '''
        categories = self.CONFIG['categories']

        # if category not in categories: # TODO 기능이 구체화 되면 다시 사용
        #     raise Exception('category must be ', categories)

        keywords = self.preprocessor.get_keywords(text)
        morphs = self.preprocessor.get_morphs(text)
        feature_vector = self.model_wrapper.similarity(text)

        return Question(text, category, answer, feature_vector, keywords, morphs)


if __name__ == '__main__':
    qm = QuestionMaker()
