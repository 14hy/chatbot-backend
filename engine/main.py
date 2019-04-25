from time import time

from engine.data.handler import ChatHandler
from engine.model.bert import Model
from engine.services.shuttle import ShuttleBus


class Engine(object):

    def __init__(self):

        self.context = None
        self.example = None
        self.vocab = None
        self.model = Model()
        self.chat_handler = ChatHandler()
        self._service_shuttle = ShuttleBus()
        #
        # self.test_mode()
        # self.preprocessor = PreProcessor()
        # input_feature = self.preprocessor.create_InputFeature(self.question,
        #                                                       self.context)
        # start, end = self.model.predict(input_feature)  # warm up.
        #
        # self.answer = self.preprocessor.pred_to_text(start, end, input_feature)
    def test_mode(self):
        self.question = '오리아나가 사고를 당해 하게 된 일은?'
        self.context = '오리아나는 한 때 살아있는 육신을 가진 호기심 많은 소녀였지만, ' \
                       '이제는 전체가 시계태엽 장치로 만들어진 놀라운 기술의 산물이다. ' \
                       '오리아나는 자운 남부지방에서 사고를 당한 후 매우 위태로운 상황에 처했고, ' \
                       '다쳐서 움직일 수 없는 신체의 부분 부분이 정교한 인공 신체로 교체되었다. ' \
                       '오리아나는 자신을 보호하는 친구 역할을 해 주는 강력한 황동 구체와 함께, ' \
                       '이제 필트오버를 비롯해 온 세상에 있는 불가사의를 자유롭게 탐험한다.'
    def test_mode_2(self):
        self.question = '임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?'
        self.context = '1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다.' \
                       ' 1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. ' \
                       '같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. ' \
                       '경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, ' \
                       '12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등' \
                       ' 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다.' \
                       ' 1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 ' \
                       '학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. 임종석은 청량리경찰서에서 ' \
                       '약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다.'

    def test(self, text):
        input_feature = self.preprocessor.create_InputFeature(text, self.context)
        start, end = self.model.predict(input_feature)
        print("***테스트:::", self.preprocessor.pred_to_text(start, end, input_feature))


    def get_context(self):
        return self.context

    def _set_context(self, context):
        self.context = context

    def match_context(self):
        if self.question is None:
            raise Exception("question is None")
        pass

    def chat_to_answer(self, chat):
        '''
        프론트로부터 질문을 받아 적절한 답변을 보냄
        :param chat: str
        :return: str
        '''
        tic = time()

        # TODO Query Feature extractor.
        query = self.chat_handler.handle_chat(chat)

        answer = self.answer_by_category(query.matched_question)
        # 함수화하기
        toc = time()
        print('*** 생성된 답변 ***\n', answer)
        print('*** 소모 시간: {}'.format(toc-tic))

        return answer

    def _text_to_feature_vectors(self, text):
        input_feature = self.preprocessor.create_InputFeature(text,
                                                              self.context)
        input_feature.show()
        feature_vectors = self.model.extract_feature_vector(input_feature, -2)
        print('*** 생성된 feature vector ***\n', feature_vectors)
        return feature_vectors

    def answer_by_category(self, matched_question):

        category = matched_question.category

        if category == 'shuttle_bus':
            return self._service_shuttle.response()
        elif category == 'talk':
            return {"mode": "talk", "response": matched_question.answer}

if __name__ == '__main__':
    main = Engine()
    print('ENGINE MAIN')