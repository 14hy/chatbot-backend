import collections
import unicodedata

from khaiii import KhaiiiApi


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
                    if start > 0:
                        substr = '##' + substr
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


def load_vocab_as_list(vocab_file):
    vocab = []
    with open(vocab_file, mode='r', encoding='utf8') as f:
        while (True):
            line = f.readline()
            line = line.strip('\n')
            if not line:
                break
            vocab.append(line)
    return vocab


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


class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):

        self.vocab = load_vocab(vocab_file)  # key - value(idx)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.khaiii_api = KhaiiiApi()

    def str_to_morphs(self, text):
        '''
        한글 형태소 분석기 khaiii
        :param text:
        :return: 형태소 단위로 띄어쓰기 된 text

        ** KhaiiiWord **
        lex : 원본의 토큰
        begin : 원본에서 토큰의 시작 위치
        morphs : KhaiiiMorph 객체들의 리스트
        *** KhaiiiMorph ***
        lex : 형태소 토큰
        tag : 품사
        '''
        output = []
        for word in self.khaiii_api.analyze(text):
            morphs = word.morphs
            for morph in morphs:
                output.append(morph.lex)

        return ' '.join(output)

    def tokenize(self, question_text) -> list:
        '''
        :param question_text: str
        :return: list of str
        '''
        tokens = []
        for token in self.basic_tokenizer.tokenize(question_text):
            token = unicodedata.normalize('NFC', token)  # 첫가끝소리 -> 소리마디 (NFD -> NFC)
            token = self.str_to_morphs(token)
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


class InputFeatures(object):
    def __init__(self,
                 unique_id,
                 input_ids,
                 input_mask=None,
                 segment_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.unique_id = unique_id

    def show(self):
        print('input_ids:', self.input_ids)
        print('input_mask:', self.input_mask)
        print('segment_ids:', self.segment_ids)
        print('unique_id:', self.unique_id)


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


class PreProcessor(object):

    def __init__(self, params):

        self.tokenizer = FullTokenizer(params['vocab_file'])
        self.vocab = load_vocab_as_list(params['vocab_file'])


    def str_to_tokens(self, text):
        '''

        :param text: str
        :return: list[str] tokenized wordpieces
        '''
        return self.tokenizer.tokenize(text)

    def tokens_to_idx(self, tokens):
        '''

        :param tokens: list of tokens
        :return: list of indexes
        '''
        output = []
        for token in tokens:
            output.append(self.vocab.index(token))
        return output

    def create_feature(self, question, context, params):

        unique_id = params['unique_id']
        max_query_length = params['max_query_length']
        max_seq_length = params['max_seq_length']

        question_text = question
        doc_tokens = _tokenize_str(context)

        token_to_original_index = []
        original_to_token_index = []
        all_doc_tokens = []

        query_tokens = self.str_to_tokens(question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        for i, token in enumerate(doc_tokens):
            original_to_token_index.append(i)
            sub_tokens = self.str_to_tokens(token)
            for sub_token in sub_tokens:
                token_to_original_index.append(i)
                all_doc_tokens.append(sub_token)

        input_ids = []
        input_mask = []
        segment_ids = []

        input_ids.append('[CLS]')
        segment_ids.append(0)
        for query in query_tokens:
            input_ids.append(query)
            segment_ids.append(0)
        input_ids.append('[SEP]')
        segment_ids.append(0)

        # context
        for doc in all_doc_tokens:
            input_ids.append(doc)
            segment_ids.append(1)
        input_ids.append('[SEP]')
        segment_ids.append(1)

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[0:max_seq_length]
            input_ids[-1] = ['SEP']
            segment_ids = segment_ids[0:max_seq_length]

        _length = len(input_ids)

        for _ in range(_length):
            input_mask.append(1)

        for _ in range(max_seq_length - _length):
            input_mask.append(0)
            segment_ids.append(0)
            input_ids.append(0)

        for i in range(len(input_ids)):
            if input_ids[i] in self.vocab:
                input_ids[i] = self.vocab.index(input_ids[i])

        # input_ids = self.tokens_to_idx(input_ids)

        feature = InputFeatures(unique_id,
                                input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids)

        return feature

    def pred_to_text(self, start, end, feature):

        pred_answer_text = ''
        for i in range(start[0], end[0] + 1):
            vocab_idx = feature.input_ids[i]
            word = self.vocab[vocab_idx]
            if '#' in word:
                pred_answer_text += word.strip('#')
            else:
                pred_answer_text += ' ' + word

        return pred_answer_text
