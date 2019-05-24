import config
from src.data.preprocessor import PreProcessor
from src.db.questions.question import Question
from src.db.questions import index as _questions
from src.model.serving import TensorServer
from sklearn.feature_extraction.text import TfidfVectorizer


class QuestionMaker(object):

    def __init__(self):
        self.CONFIG = config.QUESTION
        self.model_wrapper = TensorServer()
        self.preprocessor = PreProcessor()
        vocab = self.preprocessor.vocab[:-1]
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, min_df=0,
                                                stop_words=None, vocabulary=vocab)
        self.idf_, self.vocabulary_ = self.set_idf()

    def create_question(self, text, answer=None, category=None):
        text, removed = self.preprocessor.clean(text)

        if category not in self.CONFIG['categories']:
            raise Exception('category must be ', self.CONFIG['categories'])

        keywords = self.preprocessor.get_keywords(text=text)
        morphs = self.preprocessor.get_morphs(text=text)
        vector = self.model_wrapper.similarity(text)  # ELMO LIKE

        return Question(text, category, answer, vector, morphs, keywords=keywords)

    def set_idf(self):
        question_list = _questions.find_all()

        raw_documents = []

        for question in question_list:
            text = ' '.join(self.preprocessor.str_to_tokens(question.text))
            raw_documents.append(text)

        self.tfidf_vectorizer.fit_transform(raw_documents=raw_documents)
        idf_ = self.tfidf_vectorizer.idf_ / max(self.tfidf_vectorizer.idf_)  # 최대값으로 정규화
        return idf_, self.tfidf_vectorizer.vocabulary_

    def insert_text(self, text, answer=None, category=None):
        question = self.create_question(text, answer, category)
        return _questions.insert(question)

    def rebase(self):
        questions = _questions.find_all()

        for question in questions:

            backup = None
            orig_text = question.text
            try:
                backup = question
                question = self.create_question(text=question.text,
                                                category=question.category,
                                                answer=question.answer)
                _questions.delete_by_text(orig_text)
                _questions.insert(question)
                print('rebase: {}'.format(question.text))
            except Exception as err:
                print('rebase: ', str(err))
                if backup:
                    _questions.insert(backup)
                return


if __name__ == '__main__':
    qm = QuestionMaker()
    qm.rebase()
