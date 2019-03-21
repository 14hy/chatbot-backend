import collections
import unicodedata


def convert_by_vocab(vocab, items):
    '''

    :param vocab:
    :param items:
    :return:
    '''
    output = []
    for item in items:
        output.append(vocab[item])
    return output

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False






def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False  # _is_whitespace 에서 처리할 것임
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    '''
    :param char:
    :return:
    '''
    cp = ord(char)  # 유니코드
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BasicTokenizer(object):
    '''
    punctuation (, ; : - _ 등,) spliting, 대문자 -> 소문자 등
    1. punctuation - , : ; - _
    2. lower case
    3. ?
    4. accent?
    5. 중국어?
    '''

    def __init__(self, do_lower_case=True):

        self.do_lower_case = True

    def tokenize(self, text):
        '''

        :param text:
        :return:
        '''
        # 중국어처리뺌
        text = self._clean_text(text)
        # assert text != '' # ?
        text_split_by_whitespace = whitespace_tokenize(text)
        split_tokens = []
        for token in text_split_by_whitespace:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_split_on_punc(self, text):
        '''
        조금 쓸데 없이 어렵게 하는 거 같기도 한데..
        :param text: token split by whitespace
        :return:
        '''
        output = []
        start_new_word = True
        for char in text:
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)

        return [''.join(x) for x in output]

    def _clean_text(self, text):
        '''
        부적절한 character 삭제 및 빈칸 제거
        :param text: str
        :return: str
        '''

        output = []

        for char in text:
            # cp = 0 - NULL,
            # null 이거나, "",
            cp = ord(char)  # 유니코드 번호로 리턴
            if cp == 0 or cp == 0xfffd or _is_control(char):  # null, '', control character(EOF등) 라면, 넣지 않음
                continue
            if _is_whitespace(char):  # \t, \r \n 등 모두 띄어쓰기로 처리
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _run_strip_accents(self, text):
        '''
        :param text: str
        :return: str
        '''
        # 유니코드 정규화
        # https://ko.wikipedia.org/wiki/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C_%EC%A0%95%EA%B7%9C%ED%99%94
        text = unicodedata.normalize("NFD", text)  # normalize 하면 자모음 단위로 쪼갤 수 있게 됨
        output = []
        chars = list(text)
        for i in range(len(chars)):  #
            cat = unicodedata.category(chars[i])
            if cat == "Mn":  # “Nonspacing Mark”
                continue
            output.append(chars[i])
        return "".join(output)


class WordpieceTokenizer(object):
    '''wordpiece 임베딩에 맞는 형식으로 변환'''

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        '''
        긴 토큰부터 매칭시켜줌
        :param text: str token
        :return: a list of wordpiece tokens
        '''

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False  # unknown voca
            start = 0  # for greedy algorithm
            sub_tokens = []  # wordpiece 형식/ 단위
            # greedy algorithm finding longest token.
            while start < len(chars):
                end = len(chars)  # ?
                cur_substr = None  # ?
                while start < end:
                    substr = ''.join(chars[start:end])
                    # if start > 0:
                    #     substr = '##' + substr # TODO
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1  # 긴 것 부터 찾도록,
                if cur_substr is None:
                    # 찾지 못함
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                # 찾지 못할 경우, unknown token을 추가
                # 단어의 모든 substr을 찾을 수 있어야 함. - 헷갈렸네.
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf8') as reader:
        for token in reader:
            # token = convert_to_unicode(reader.readline())
            token = token.strip()
            vocab[token] = index
            index += 1

    print('---loaded vocab from {}---'.format(vocab_file))
    print('num of vocab:', len(vocab))
    return vocab

class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):

        self.vocab = load_vocab(vocab_file)  # key - value(idx)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)


    def tokenize(self, question_text) -> list:
        '''
        :param question_text: str
        :return: list of str
        '''
        tokens = []
        for token in self.basic_tokenizer.tokenize(question_text):
            token = unicodedata.normalize('NFC', token)  # 첫가끝소리 -> 소리마디 (NFD -> NFC)
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                tokens.append(sub_token)
                # ex)
                # <class 'list'>: ['유', '##전', '##자', '중', '##복', '##에', '의한', 'd', '##na', '염', '##기', '##서', '##열', '##은',
                # '어떤', '순', '##서를', '바', '##꿔', ',', '본', 'd', '##na', '##와', '다른', '단', '##백', '##질']
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self):
        pass

    def load_vocab(self):
        self.vocab = collections.OrderedDict()

        with open(file=self.args.vocab_file, mode='r') as file:
            idx = 0
            for line in file:
                self.vocab[line] = idx
                idx += 1

    def tokenize_to_doc_tokens(self, context):
        def _tokenize_str(text):
            '''
            text 를 받아, 전처리 한 후, 토큰들로 바꿈
            :param text: string
            :return: list of tokens
            '''
            tokens = []
            prev_is_whitespace = True

            for char in text:
                if _is_whitespace(char):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        tokens.append(char)
                    else:
                        tokens[-1] += char
                    prev_is_whitespace = False

            return tokens
        return _tokenize_str(context)
