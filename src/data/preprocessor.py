import config
from src.data.mytokenization import FullTokenizer
from src.utils import Singleton


def load_vocab_as_list(vocab_file):
    vocab = []
    with open(vocab_file, mode='r', encoding='utf8') as f:
        while True:
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
                 input_masks=None,
                 segment_ids=None,
                 doc_tokens=None,
                 tok_to_orig_map=None):
        self.tok_to_orig_map = tok_to_orig_map
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.doc_tokens = doc_tokens

    def __str__(self):
        return 'InputFeature\n input_ids:{}\n input_masks:{}\n segement_ids:{}'.format(self.input_ids, self.input_masks,
                                                                                       self.segment_ids)


class PreProcessor(metaclass=Singleton):

    def __init__(self):

        self.CONFIG = config.PREPROCESS

        self.tokenizer = FullTokenizer(self.CONFIG['vocab_file'], use_morphs=self.CONFIG['use_morphs'])
        self.vocab = load_vocab_as_list(self.CONFIG['vocab_file'])

    def str_to_tokens(self, text):
        '''

        :param text: str
        :return: list[str] tokenized wordpieces
        '''
        return self.tokenizer.tokenize(text)

    def get_morphs(self, text):
        return self.tokenizer.text_to_morphs(text)

    def get_keywords(self, text):
        return self.tokenizer.get_keywords(text, self.CONFIG['keywords_tags'])

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

        :param query_text: str, 질문
        :param context: str, Squad일 때 사용
        :return: InputFeatures object

        context is not None:
        input_ids: [CLS] query_text [SEP] context [SEP] [PAD] ...
        segment_ids: [0] [0] [0] [0] [0] [1] [1] [1] [1] [0] ...
        input_masks:  [1] [1] [1] [1] [1] [1] [1] [1] [1] [0] ...
        context is None:
        input_ids: [CLS] query_text [SEP] [PAD] ...
        segment_ids: [0] [0] [0] [0] [0] [0] [0] ...
        input_masks:  [1] [1] [1] [1] [1] [1] [0] ...
        '''
        if not context:
            max_query_length = self.CONFIG['max_query_length-similarity']
            max_seq_length = max_query_length
        else:
            max_query_length = self.CONFIG['max_query_length-search']
            max_seq_length = self.CONFIG['max_seq_length-search']

        tok_to_original_index = []
        orig_to_token_idx = []
        all_doc_tokens = []
        doc_tokens = None
        tok_to_orig_map = None

        query_tokens = self.str_to_tokens(query_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        input_ids = []
        input_masks = []
        segment_ids = []

        input_ids.append('[CLS]')
        segment_ids.append(0)
        for query in query_tokens:
            input_ids.append(query)
            segment_ids.append(0)
        input_ids.append('[SEP]')
        segment_ids.append(0)

        if context is not None:
            # doc_tokens = self.tokenizer.tokenize_to_doc_tokens(context)
            doc_tokens = context.split()
            tok_to_orig_map = {}

            j = 0
            for i, token in enumerate(doc_tokens):
                orig_to_token_idx.append(i)
                sub_tokens = self.str_to_tokens(token)

                for sub_token in sub_tokens:
                    tok_to_original_index.append(i)
                    tok_to_orig_map[len(input_ids) + j] = tok_to_original_index[len(tok_to_original_index) - 1]
                    all_doc_tokens.append(sub_token)
                    j += 1

            for doc in all_doc_tokens:
                input_ids.append(doc)
                segment_ids.append(1)
            input_ids.append('[SEP]')
            segment_ids.append(1)

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[0:max_seq_length]
            input_ids[-1] = '[SEP]'
            segment_ids = segment_ids[0:max_seq_length]

        _length = len(input_ids)

        for _ in range(_length):
            input_masks.append(1)

        for _ in range(max_seq_length - _length):
            input_masks.append(0)
            segment_ids.append(0)
            input_ids.append(0)  # 0 = [PAD]

        for i in range(len(input_ids)):
            if input_ids[i] in self.vocab:
                input_ids[i] = self.vocab.index(input_ids[i])

        # input_ids = self.tokens_to_idx(input_ids)

        feature = InputFeatures(input_ids=input_ids,
                                input_masks=input_masks,
                                segment_ids=segment_ids,
                                doc_tokens=doc_tokens,
                                tok_to_orig_map=tok_to_orig_map)

        return feature

    def idx_to_orig(self, start, end, feature):

        tok_to_orig_map = feature.tok_to_orig_map
        doc_tokens = feature.doc_tokens

        try:

            orig_start = tok_to_orig_map[start]
            orig_end = tok_to_orig_map[end]
            orig_text = doc_tokens[orig_start:orig_end + 1]
            if orig_start > orig_end:
                orig_text = doc_tokens[orig_end:orig_start + 1]
                # 모델 성능 때문에 start가 더 높게 나오는 경우가 있음
            orig_text = ' '.join(orig_text)
            return self.clean_tags(orig_text)
        except Exception as err:
            return False

    def clean_tags(self, text):
        # 조사 제거
        final_morphs = self.tokenizer.text_to_morphs(text)
        tags = self.CONFIG['clean_tags']
        # https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY/edit#gid=589544265

        n = -1
        deletion_len = 0
        while True:
            found = False
            output = final_morphs['text'].split()
            last_morph = output[n]
            tag = final_morphs[last_morph]
            for t in tags:
                if t in tag:
                    deletion_len += len(last_morph)
                    found = True
                    break
            if not found:
                break
            n -= 1
            if len(output) < -n:
                break
        if deletion_len != 0:
            text = text[:-deletion_len]

        return text

    def clean(self, chat):
        chat, removed = self.tokenizer.clean_chat(chat)

        return chat, removed


if __name__ == "__main__":
    prep = PreProcessor()
