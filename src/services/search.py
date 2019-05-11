import config
from src.data.preprocess import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer

from src.model.serving import TensorServer
from src.utils import Singleton
from src.db.contexts import index as contexts

CONFIG = config.SEARCH


class Search(metaclass=Singleton):

    def __init__(self):
        self.tfidf_matrix = None
        self.contexts_list = None

        self.tfidf_vectorizer = TfidfVectorizer(stop_words=None,
                                                sublinear_tf=CONFIG['sublinear_tf'])
        self.preprocessor = PreProcessor()
        self.set_contexts_list()
        self.set_tfidf_matrix()
        self.tensor_server = TensorServer()

    def response(self, chat):
        # context TF IDF 로 찾기
        context, score = self.get_context(chat)
        if score == 0:
            return None, None, None
        text = context['text']
        answer = self.tensor_server.search(chat, text)

        return answer, text, score

    def set_tfidf_matrix(self):
        text_list = list(map(lambda x: ' '.join(self.preprocessor.get_keywords(x['text'])),
                             self.contexts_list))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_list).todense().tolist()

    def set_contexts_list(self):
        self.contexts_list = list(contexts.find_all())

    def get_context(self, chat):
        chat = ' '.join(self.preprocessor.get_keywords(chat))
        chat_tfidf = self.tfidf_vectorizer.transform([chat]).todense().tolist()[0]
        num_context = len(self.tfidf_matrix)

        max_score = 0
        score = 0
        max_context = -999

        for i in range(num_context):
            context_tfidf = self.tfidf_matrix[i]
            num_context_voca = len(context_tfidf)
            for j in range(num_context_voca):
                score += chat_tfidf[j] * context_tfidf[j]
            if score >= max_score:
                max_score = score
                max_context = i
            score = 0

        return self.contexts_list[max_context], max_score


if __name__ == '__main__':
    test = Search()
    print(test.get_context('하냥이를 제작한 창업 동아리는?'))
    print(test.response('하냥이가 적용 된 곳은?'))
