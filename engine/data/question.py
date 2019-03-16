import config
from engine.data.preprocess import PreProcessor
from engine.model.bert import Model
from khaiii import KhaiiiApi

class Question(object):
    def __init__(self, text, category, answer, feature_vector, morphs, object_id=None):
        '''

        :param text: original question text
        :param category:
        :param answer:
        :param feature_vector:
        :param morphs: KhaiiiWord object
        '''
        self.object_id = object_id
        self.category = category
        self.answer = answer
        self.text = text
        self.morphs = morphs
        self.feature_vector = feature_vector

class QuestionMaker(object):
    def __init__(self):

        self.DEFAULT_CONFIG = config.DEFAULT_CONFIG

        self.preprocessor = PreProcessor()

        self.bert_model = Model()

        self.khaiii_api = KhaiiiApi()


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
        feature_vector = self.bert_model.extract_feature_vector(input_feature, layers=-1)

        morphs = self.khaiii_api.analyze(text)

        return Question(text, category, answer, feature_vector, morphs)

if __name__ == '__main__':
    qm = QuestionMaker()
    print(qm.create_question('안녕하세요, 테스트 입니다!', '셔틀', answer='넵.').feature_vector)
    print(qm.create_question('안녕하세요, 테스트 입니다!', '셔틀', answer='넵.').feature_vector.shape)