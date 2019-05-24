import config
from src.data.preprocessor import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer

from src.model.serving import TensorServer
from src.utils import Singleton
from src.db.contexts import index as contexts


class Search(metaclass=Singleton):

    def __init__(self):
        self.tfidf_matrix = None
        self.contexts_list = None
        self.CONFIG = config.SEARCH

        self.tfidf_vectorizer = TfidfVectorizer(stop_words=None,
                                                sublinear_tf=self.CONFIG['sublinear_tf'])
        self.preprocessor = PreProcessor()
        self.set_context()
        self.set_tfidf_matrix()
        self.tensor_server = TensorServer()

    def response(self, chat):
        # context TF IDF 로 찾기
        output = self.find_context(chat)
        context = output['context-1']
        score = output['score-1']
        if score == 0:
            return None, None
        answer = self.tensor_server.search(chat, context)

        return answer, output

    def response_with_subject(self, _chat, _subject):
        context = contexts.find_by_subject(_subject=_subject)
        context = context['text']
        answer = self.tensor_server.search(chat=_chat, context=context)

        return answer, context

    def response_with_context(self, _chat, _context):
        answer = self.tensor_server.search(chat=_chat, context=_context)

        return answer

    def response_with_id(self, _chat, _id):
        context = contexts.find_by_id(_id=_id)['text']

        return self.tensor_server.search(chat=_chat, context=context), context

    def set_tfidf_matrix(self):
        text_list = list(map(lambda x: ' '.join(self.preprocessor.get_keywords(x['text'])),
                             self.contexts_list))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_list).todense().tolist()

    def set_context(self):
        self.contexts_list = list(contexts.find_all())

    def find_context(self, chat):
        chat = ' '.join(self.preprocessor.get_keywords(chat))
        chat_tfidf = self.tfidf_vectorizer.transform([chat]).todense().tolist()[0]
        num_context = len(self.tfidf_matrix)

        score = 0

        ordered_list = []

        output = {
            'context_subject-1': None,
            'context_subject-2': None,
            'context_subject-3': None,
            'context-1': None,
            'context-2': None,
            'context-3': None,
            'score-1': None,
            'score-2': None,
            'score-3': None
        }

        for i in range(num_context):
            context_tfidf = self.tfidf_matrix[i]
            num_context_voca = len(context_tfidf)
            for j in range(num_context_voca):
                score += chat_tfidf[j] * context_tfidf[j]
            ordered_list.append((i, score))
            score = 0

        ordered_list = sorted(ordered_list, key=lambda x: x[1], reverse=True)
        for i in range(self.CONFIG['max_context_num']):
            output['context_subject-{}'.format(i + 1)] = self.get_context(ordered_list[i][0])['subject']
            output['score-{}'.format(i + 1)] = ordered_list[i][1]
            output['context-{}'.format(i + 1)] = self.get_context(ordered_list[i][0])['text']

        return output

    def get_context(self, idx):
        return self.contexts_list[idx]


if __name__ == '__main__':
    test = Search()
    # print(test.get_context('하냥이를 제작한 창업 동아리는?'))
    # print(test.response('하냥이가 적용 된 곳은?'))
    # print(test.response('하냥이를 제작한 창업 동아리는?'))
    # print(test.response('팀프로젝트를 할 만 한 곳?'))
    # print(test.response('아고라가 뭐에요?'))
    print(test.find_context('아고라가 뭐에요?'))
