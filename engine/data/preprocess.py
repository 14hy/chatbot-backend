import config
from engine.tokenization import FullTokenizer
from engine.utils import Singleton


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


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask=None,
                 segment_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
    def show(self):
        print('input_ids:', self.input_ids)
        print('input_mask:', self.input_mask)
        print('segment_ids:', self.segment_ids)


class PreProcessor(metaclass=Singleton):

    def __init__(self):

        self.DEFAULT_CONFIG = config.DEFAULT_CONFIG

        self.tokenizer = FullTokenizer(self.DEFAULT_CONFIG['vocab_file'])
        self.vocab = load_vocab_as_list(self.DEFAULT_CONFIG['vocab_file'])


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

    def create_InputFeature(self, query_text, context=None):
        '''

        :param query_text:
        :param context:
        :param params:
        :return:

        context is not None:
        input_ids: [CLS] query_text [SEP] context [SEP] [PAD] ...
        segment_ids: [0] [0] [0] [0] [0] [1] [1] [1] [1] [0] ...
        input_mask:  [1] [1] [1] [1] [1] [1] [1] [1] [1] [0] ...
        context is None:
        input_ids: [CLS] query_text [SEP] [PAD] ...
        segment_ids: [0] [0] [0] [0] [0] [0] [0] ...
        input_mask:  [1] [1] [1] [1] [1] [1] [0] ...
        '''

        max_query_length = self.DEFAULT_CONFIG['max_query_length']
        max_seq_length = self.DEFAULT_CONFIG['max_seq_length']


        token_to_original_index = []
        original_to_token_index = []
        all_doc_tokens = []

        query_tokens = self.str_to_tokens(query_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

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

        if context is not None:
            doc_tokens = self.tokenizer.tokenize_to_doc_tokens(context)

            for i, token in enumerate(doc_tokens):
                original_to_token_index.append(i)
                sub_tokens = self.str_to_tokens(token)
                for sub_token in sub_tokens:
                    token_to_original_index.append(i)
                    all_doc_tokens.append(sub_token)

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

        feature = InputFeatures(input_ids=input_ids,
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