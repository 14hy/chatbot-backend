import config
from engine.data.preprocess import PreProcessor
from engine.model.bert import Model

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

class QuestionMaker(object):

    def __init__(self):

        self.DEFAULT_CONFIG = config.DEFAULT_CONFIG

        self.preprocessor = PreProcessor()

        self.bert_model = Model()

    def create_question(self, text, category=None, answer=None):
        '''

        :param text: 질문
        :param category: '셔틀', '학식', '잡담', '학사행정', '검색' ...
        :param answer: 정답/ None
        :return: Question object
        '''
        categories = self.DEFAULT_CONFIG['categories']

        # if category not in categories: # TODO 기능이 구체화 되면 다시 사용
        #     raise Exception('category must be ', categories)

        input_feature = self.preprocessor.create_InputFeature(text)
        keywords = self.preprocessor.get_keywords(text)
        morphs = self.preprocessor.str_to_morphs(text)
        feature_vector = self.bert_model.extract_feature_vector(input_feature)
        print(morphs)

        return Question(text, category, answer, feature_vector, keywords, morphs)


if __name__ == '__main__':
    qm = QuestionMaker()
