import collections
import os, sys
import json
import random
from tqdm import tqdm
import tensorflow as tf


def is_whitespace(c):
    # 공백이 있는가?
    if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:  # 아스키코드 ' '
        return True
    else:
        return False

class Example(object):
    def __init__(self, qas_id, question_text, doc_tokens,
                 orig_answer_text = None, start= None, end = None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start = start
        self.end = end
    
    def show(self):
        print('qas_id:', self.qas_id)
        print('question_text:', self.question_text)
        print('doc_tokens:', self.doc_tokens)
        print('orig_answer_text:', self.orig_answer_text)
        print('start:', self.start)
        print('end:', self.end)


class InputFeatures(object):
    def __init__(self, unique_id,
                example_index,
                tokens,
                token_to_orig_map,
                input_ids,
                doc_span_index = None,
                token_is_max_context = None,
                input_mask = None,
                segment_ids = None,
                start_position = None,
                end_position = None,
                is_impossible=None):
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.doc_span_index = doc_span_index
        self.example_index = example_index
        self.unique_id = unique_id
        self.is_impossible = is_impossible
    def show(self):
        print('tokens:', self.tokens)
        print('token_to_orig_map:', self.token_to_orig_map)
        print('token_is_max_context:', self.token_is_max_context)
        print('input_ids:', self.input_ids)
        print('input_mask:', self.input_mask)
        print('segment_ids:', self.segment_ids)
        print('start_position:', self.start_position)
        print('end_position:', self.end_position)
        print('doc_span_index:', self.doc_span_index)
        print('example_index:', self.example_index)
        print('unique_id:', self.unique_id)
        print('is_impossible:', self.is_impossible)

def tokenize(text):
    '''

    :param text: string
    :return: list of tokens
    '''
    tokens = []
    prev_is_whitespace = True

    for char in text:
        if is_whitespace(char):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(char)
            else:
                tokens[-1] += char
            prev_is_whitespace = False

    return tokens