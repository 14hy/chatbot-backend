from pprint import pprint

from khaiii import KhaiiiApi
from mecab import MeCab

from src.data.tokenization import *
from src.data.tokenization import _is_punctuation


class FullTokenizer(FullTokenizer):

    def __init__(self, vocab_file,
                 do_lower_case=True,
                 use_morphs=False,
                 log=False, stop_words_file=None,
                 sub_file=None):
        self.use_morphs = use_morphs
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, stop_words_file=stop_words_file,
                                              sub_file=sub_file)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                      use_morphs=self.use_morphs)

        self.khaiii_api = KhaiiiApi()
        self.mecab_api = MeCab()
        self._khaiii_memoization = {}
        self.log = log
        print('*** running with my tokenizer ***')
        print('*** vocab file: {}***'.format(vocab_file))

    def clean_chat(self, chat):

        removed = []

        chat, r = self.basic_tokenizer.clean_punctuations(chat)
        chat, r2 = self.basic_tokenizer.clean_stop_words(chat)
        chat = self.basic_tokenizer.sub(chat)
        removed.extend(r + r2)

        return chat, removed

    def text_to_morphs(self, text):
        morphs = {}
        output = []

        tokens = self.mecab_api.pos(text)
        for text in tokens:
            word = text[0]
            tag = text[1]
            if '+' in tag:
                if word in self._khaiii_memoization:
                    khaiii = self._khaiii_memoization[word]
                else:
                    khaiii = self.khaiii_api.analyze(word)[0].morphs
                    self._khaiii_memoization[word] = khaiii
                for each in khaiii:
                    output.append(each.lex)
                    morphs[each.lex] = each.tag
            else:
                morphs[word] = tag
                output.append(word)
        output = ' '.join(output)
        morphs['text'] = output
        if self.log:
            pprint(morphs)
        return morphs

    def get_keywords(self, text, tag_NN):
        """
        질문으로 부터 체언을 뽑아 키워드로 생각하고 리턴
        :param tag_NN: 뽑을 키워드의 태그
        :param text: str,
        :return: list, 키워드
        """
        keywords = []
        # Mecab 에서 제공하는 체언 품사들 # TODO 체언 말고도 쓸만한 품사가 있을 지?

        tokens = self.mecab_api.pos(text)
        for token in tokens:
            word = token[0]
            tag = token[1]
            if tag in tag_NN:
                keywords.append(word)

        return keywords

    def tokenize(self, text):
        """
        Basic tokenizer -> 형태소 분석 -> Word-piece tokenizer
        :param text: str, 질문
        :return: list, 토큰들
        """
        split_tokens = []
        if self.use_morphs:
            text = self.text_to_morphs(text)['text']
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens


class WordpieceTokenizer(WordpieceTokenizer):

    def __init__(self, vocab, unk_token="[UNK]",
                 max_input_chars_per_word=200,
                 use_morphs=False):
        self.use_morphs = use_morphs
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of chat into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
        input = "unaffable"
        output = ["un", "##aff", "##able"]

        Args:
        chat: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

        Returns:
        A list of wordpiece tokens.
        """
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0 and not self.use_morphs:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BasicTokenizer(BasicTokenizer):

    def __init__(self, do_lower_case=True, stop_words_file=None, sub_file=None):
        self.do_lower_case = do_lower_case
        self.stop_words = self.load_stop_words(stop_words_file)
        self.sub_dic = self.load_sub(sub_file)

    def load_stop_words(self, file):
        if file is None:
            return None
        stop_words = []

        with open(file, mode='r') as f:
            for line in f:
                convert_to_unicode(line)
                stop_words.append(line.strip())

        return stop_words

    def load_sub(self, sub_file):
        if sub_file is None:
            return None
        sub_dic = {}
        with open(sub_file, mode='r', encoding='utf-8') as f:
            for line in f:
                line = convert_to_unicode(line).strip()
                line = line.split(',')
                sub_dic[line[0]] = line[1]
        return sub_dic

    def sub(self, text):
        text = text.split()
        output = []

        for t in text:
            if t in self.sub_dic.keys():
                output.append(self.sub_dic[t])
            else:
                output.append(t)

        return ' '.join(output)

    def is_stop_words(self, text):

        for word in self.stop_words:
            if word == text:
                return True

        return False

    def clean_punctuations(self, text):

        text = list(text)
        output = []
        removed = []

        for char in text:
            if _is_punctuation(char):
                removed.append(char)
            else:
                output.append(char)
        output = ' '.join(whitespace_tokenize(''.join(output)))
        return output, removed

    def clean_stop_words(self, text):

        text = text.split()
        output = []
        removed = []

        for t in text:
            if self.is_stop_words(t):
                pass
            else:
                output.append(t)

        return ' '.join(output), removed


if __name__ == '__main__':
    test = BasicTokenizer(True, './stop_words.txt')
