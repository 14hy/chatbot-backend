import collections
import json
import math
import os
import re
import sys
import time
import tensorflow as tf
import numpy as np

import config
from src.data.preprocessor import PreProcessor


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])

        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


class BertConfig(object):
    def __init__(self,
                 vocab_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    def read_from_json_file(self, json_file):
        with open(json_file, mode='r') as file:
            config = json.load(file)
            print('*** BERT CONFIGURATION JSON ***')
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_act = config['hidden_act']
        self.intermediate_size = config['intermediate_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.max_position_embeddings = config['max_position_embeddings']
        self.type_vocab_size = config['type_vocab_size']
        self.initializer_range = config['initializer_range']

        print('vocab_size:', self.vocab_size)
        print('hidden_size:', self.hidden_size)
        print('num_hidden_layers:', self.num_hidden_layers)
        print('num_attention_heads:', self.num_attention_heads)
        print('hidden_act:', self.hidden_act)
        print('intermediate_size:', self.intermediate_size)
        print('hidden_dropout_prob:', self.hidden_dropout_prob)
        print('attention_probs_dropout_prob:', self.attention_probs_dropout_prob)
        print('max_position_embeddings:', self.max_position_embeddings)
        print('type_vocab_size:', self.type_vocab_size)
        print('initializer_range:', self.initializer_range)


def gelu(x):
    '''Gaussian Error Linear Unit'''
    '''
    RELU 보다 부드럽다고 한다.
    별 차이는 없고, 음수에서 0이 아니고 어느정도 값이 존재함
    양수에서 초반에는 Relu보다 값이 작고 결국에는 같아진다.
    '''

    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_shape(tensor, expected_rank=None):
    shape = tensor.shape.as_list()

    assert len(shape) == expected_rank

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size,
                     initializer_range,
                     word_embedding_name='word_embeddings',
                     use_one_hot_embeddings=False):
    '''
    id<int> => embedding feature_vector<float>[embedding size]

    :param input_ids: int Tensor of shape [batch_size, seq_length]
    :param vocab_size: int.
    :param embedding_size: int -> same as hidden size
    :param initializer_range: float 임베딩 초기화 범위.
    :param word_embedding_name: str
    :param use_one_hot_embeddings: True -> one-hot False -> tf.gather() # ?
    :return: float Tensor of shape [batch_size, seq_length, embedding_size].
    '''
    embedding_output = None
    embedding_table = None

    if len(get_shape(input_ids, 2)) == 2:
        input_ids = tf.expand_dims(input=input_ids, axis=-1)  # axis에 1차원 추가 => [batch_size, seq_length, 1]

    assert len(get_shape(input_ids, 3))

    embedding_table = tf.get_variable(name=word_embedding_name,
                                      shape=(vocab_size, embedding_size),
                                      initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    # truncated normal distribution - 너무 작거나, 큰 값으로 초기화 하기 않기 때문에 vanishing, expand..등의 문제 해결..
    # range는 0.02정도의 매우 작은 값을 사용했음 ( default)

    flat_input_ids = tf.reshape(input_ids, [-1])  # flattend input_ids. - gather의 indices parameter가 1차원 배열을 받기 때문.

    embedding_output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape(input_ids, 3)

    embedding_output = tf.reshape(embedding_output,
                                  input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return embedding_output, embedding_table


def embedding_postprocessor(input_tensor, use_token_type, token_type_ids, token_type_vocab_size,
                            token_type_embedding_name, use_position_embeddings, position_embedding_name,
                            initializer_range, max_position_embeddings, dropout_prob):
    '''
    임베딩된 token + segment + position sum.
    :param input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size], 임베딩된 ids
    :param use_token_type: bool, ??????????????????????????
    :param token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length] # 기본 값은 [0,] 이므로 True 여도 영향X
    :param token_type_vocab_size: int, type voca size.
    :param token_type_embedding_name: str, token type embedding name.
    :param use_position_embeddings: bool,
    :param position_embedding_name: str, postion embedding name
    :param initializer_range:  float, default 0.02
    :param max_position_embeddings: int, Maximum seq length와 같겠지?
    :param dropout_prob: float. (final output tensor)
    :return: Tensor same shape as input_tensor
    '''

    input_shape = get_shape(input_tensor, 3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]  # 임베딩 사이즈,

    output = input_tensor  # 더해 나갈 것임

    if use_token_type:  # segment부터 시작
        assert token_type_ids != None

        token_type_table = tf.get_variable(name=token_type_embedding_name,
                                           shape=(token_type_vocab_size, width),  # width가 같아야 더 할 수 있을 것
                                           initializer=tf.truncated_normal_initializer(initializer_range))

        flat_token_type_ids = tf.reshape(token_type_ids, [-1])  # [batch_size * seq_length]
        one_hot_ids = tf.one_hot(flat_token_type_ids,
                                 depth=token_type_vocab_size)  # [batch_size * seq_length, token_type_vocab_size]
        # type voca는 작기 때문에, One hot encoding 사용해도 성능이 괜찮을 것
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        # [batch_size * seq_length, token_type_vocab_size] * [token_type_vocab_size, width] = [batch_size * seq_length, width]
        # One-hot * embedding table
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings  # ids + segment

    if use_position_embeddings:
        assert seq_length <= max_position_embeddings

        full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                   shape=(max_position_embeddings, width),
                                                   initializer=tf.truncated_normal_initializer(
                                                       stddev=initializer_range))

        position_embeddings = tf.slice(full_position_embeddings, begin=[0, 0], size=[seq_length, -1])
        # width는 그대로, max_position_embeddings 는 seq길이 만큼만 짜름.
        # full_position_embeddings, [max_position_embeddings, width] => [seq_length, width]

        position_broadcast_shape = [1]  # 모든 배치에 대해 같은 값을 더할 것, broadcasting을 위한 reshape 모양 배열
        position_broadcast_shape.extend([seq_length, width])  # [1, seq_length, width]
        position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
        # position_embeddings = [seq_length, width] -> [1, seq_length, width]

        output += position_embeddings  # ids + segment + position.

    # assert output.shape.as_list() == [batch_size, seq_length, width]

    # Regularization norm & dropout(0.1). # Layer norm을 써야함, !! NOT BATCH norm.
    output = tf.contrib.layers.layer_norm(inputs=output,
                                          begin_norm_axis=-1,  # default 1,
                                          begin_params_axis=-1,  # default -1 이긴 하지만.
                                          scope=None)  # TODO 확실 한 지? - None -> LayerNorm이 기본값인듯, 맞는 듯.
    output = tf.layers.dropout(output, rate=dropout_prob)  # 원래는 tf.nn.
    return output


def create_attention_mask_from_input_mask(input_ids, input_mask):
    '''
    2D 마스크로부터 3D 어텐션 마스크 생성
    :param input_ids: 2D or 3D Tensor of shape [batch_size, seq_length]
    :param input_mask: [batch_size, to_seq_length]
    :return: Tensor of shape [batch_size, seq_length, seq_length]
    '''

    from_shape = get_shape(input_ids, 2)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape(input_mask, 2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, to_seq_length]), tf.float32)
    # element wise 곱셈을 위한 int -> float 캐스팅

    broadcast_ones = tf.ones(shape=(batch_size, from_seq_length, 1), dtype=tf.float32)

    mask = broadcast_ones * to_mask
    # broadcasting을 통한 값 복사... 이렇게 복잡하게 해야되나?
    return mask


def reshape_to_matrix(input_tensor):
    '''
    rank-2 로 reshape.
    :param input_tensor: 
    :param input_shape: 
    :return: 
    '''
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


# multi-headed attention
def attention_layer(from_tensor, to_tensor, attention_mask, num_attention_heads, size_per_head,
                    attention_probs_dropout_prob, initializer_range, do_return_2d_tensor, batch_size, from_seq_length,
                    to_seq_length):
    '''

    from_tensor 와 to_tensor 가 같다면 self attention이고, bert에서는 encoder만을 사용하므로...
    from_tensor = to_tensor = input_layer = [batch_size * seq_length, hidden_size]

    1. from_tensor 를 query로 project.
    2. to_tensor 를 key, value로 project
    -> query, key 의 dimension은 dk, value의 경우 dv.
    3. query, key scaled dot product. ( attention 과정) 둘이같으니까 self, -> why self attention 부분.
    4. softmax => value 에 대한 attention 확률 값으로.
    5. value 에 곱해서 가중치를 줌.
    6. 가중 합 된(attention된) 레이어 리턴

    :param from_tensor: [batch_size, seq_length, size_per_head]
    :param to_tensor: [batch_size, seq_length, size_per_head]
    :param attention_mask: [batch_size, seq_length, seq_length] # 여기선 mask가 모두 1이었었음.
    :param num_attention_heads: int, 논문에서 h.
    :param size_per_head: int, seq_length / num_heads.
    :param attention_probs_dropout_prob: float.
    :param initializer_range: float.
    :param do_return_2d_tensor: bool. [batch_size, seq_length, seq_length] -> [batch_size * seq_length, seq_length]
    :param batch_size: int
    :param from_seq_length: int
    :param to_seq_length: int
    :return: float Tensor of shape [batch_size, from_seq_length, num_attention_heads * size_per_head]
    or [batch_size * seq_length, seq_length] (do_return_2d_tensor = True)
    '''

    from_shape = get_shape(from_tensor, 2)
    to_shape = get_shape(to_tensor, 2)

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # rank-3 -> rank-2 로 변경,
    # [batch_size, seq_length, size_per_head] => [batch_size * seq_length, size_per_head]
    # [B, F or T, H] => [B * F or T, H]
    # from_tensor_2d = reshape_to_matrix(from_tensor)
    # to_tensor_2d = reshape_to_matrix(to_tensor)
    # 필ㄹ요없음,
    from_tensor_2d = from_tensor
    to_tensor_2d = to_tensor

    # projections
    # query, [B * F, N * H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,  # width
        activation=None,
        name="query",
        kernel_initializer=tf.truncated_normal_initializer(initializer_range))

    # key, [B * T, N * H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=None,
        name="key",
        kernel_initializer=tf.truncated_normal_initializer(initializer_range))

    # value, [B * T, N * H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=None,
        name="value",
        kernel_initializer=tf.truncated_normal_initializer(initializer_range))

    # why transpose ?
    # `query_layer` = [B, N, F, H]
    query_layer = tf.reshape(
        query_layer, [batch_size, from_seq_length, num_attention_heads, size_per_head])
    query_layer = tf.transpose(query_layer, [0, 2, 1, 3])
    # `key_layer` = [B, N, T, H]
    key_layer = tf.reshape(
        key_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])
    key_layer = tf.transpose(key_layer, [0, 2, 1, 3])
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)  # dot-product,
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))  # scaling

    if attention_mask is not None:
        # attention mask = [B, F, T] => [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0  # mask 1 => 0, 0=> -10000.0
        attention_scores += adder  # 0인 position 사실상 제거.

    # softmax.
    # attention_scores = [B, N, F, T]
    attention_probs = tf.nn.softmax(logits=attention_scores,
                                    axis=-1)
    # dropout. - ? 여기서?
    attention_probs = tf.layers.dropout(inputs=attention_probs,
                                        rate=attention_probs_dropout_prob)

    # value layer.= [B * T, N * H] => [B, T, N, H]
    # = > [B, N, T, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(context_layer,
                                   [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(context_layer,
                                   [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor
    else:
        return tf.reshape(output_tensor, shape=orig_shape_list)


def tranformer_model(input_tensor, attention_mask, hidden_size, num_hidden_layers, num_attention_heads,
                     intermediate_size, intermediate_act_fn, hidden_dropout_prob, attention_probs_dropout_prob,
                     initializer_range, do_return_all_layers):
    '''
    Original Transformer Encoder Implementation
    :param input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
    :param attention_mask: float Tensor of shape [batch_size, seq_length, seq_length]
    :param hidden_size: int = embedding_size (for residual)
    :param num_hidden_layers: int num of transformer layer (기본 12, 원래 트랜스포머는 기본 6.)
    :param num_attention_heads: int (논문에서 h)
    :param intermediate_size: int feed_forward layer ? # feed-foward layer hidden size.
    :param intermediate_act_fn: activation function. # default : gelu
    :param hidden_dropout_prob: float
    :param attention_probs_dropout_prob: float
    :param initializer_range: float
    :param do_return_all_layers: bool.
    :return: Transformer's final layer. float Tensor of shape [batch_size, seq_length, hidden_size]
    '''

    assert hidden_size % num_attention_heads == 0

    attention_head_size = int(hidden_size / num_attention_heads)

    input_shape = get_shape(input_tensor, 3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    assert input_width == hidden_size  # residual sum 을 위해.

    # prev_output = [batch_size * seq_length, hidden_size]
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []

    for layer_idx in range(num_hidden_layers):  # stack layers.
        with tf.variable_scope('layer_%d' % layer_idx):
            layer_input = prev_output

            with tf.variable_scope('attention'):
                attention_heads = []
                with tf.variable_scope('self'):  # self attention
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length
                    )
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:  # TODO 문장이 하나 일 때.
                    attention_output = attention_heads[0]
                else:
                    # TODO sequence가 여러개 일 때 발생 한다는 데,
                    # h 개의 attention들을 합치는 상황인지 ?
                    # 아직 어떤 상황인지 모르겠음,
                    # 다른 attention이 존재 한다면, self-attention head에 concat.
                    # 한 다음에 projection 할 것임.
                    # 차원이 어떻게 변하는 지?
                    attention_output = tf.concat(attention_heads, axis=-1)
                    print('len(attention_heads) != 1')  # TODO 발생 시 확인.
                    sys.exit()

                with tf.variable_scope('output'):
                    # attention_output = [batch_size * seq_length, hidden_size]
                    attention_output = tf.layers.dense(
                        inputs=attention_output,  # [batch_size * seq_length, seq_length]
                        units=hidden_size,
                        kernel_initializer=tf.truncated_normal_initializer(initializer_range)
                    )
                    attention_output = tf.layers.dropout(
                        inputs=attention_output,
                        rate=hidden_dropout_prob
                    )
                    attention_output = tf.contrib.layers.layer_norm(
                        inputs=attention_output + layer_input,  # residual sum
                        begin_norm_axis=-1,  # default 1,
                        begin_params_axis=-1,  # default -1 이긴 하지만.
                        scope=None)
            # feed foward layer 시작
            with tf.variable_scope('intermediate'):
                # intermediate_output = [batch_size * seq_length, intermediate_size]
                intermediate_output = tf.layers.dense(
                    inputs=attention_output,  # [batch_size * seq_length, hidden_size]
                    units=intermediate_size,
                    activation=intermediate_act_fn,  # default : gelu. 유일하게 activation function이 있음 - why?
                    kernel_initializer=tf.truncated_normal_initializer(initializer_range))
            with tf.variable_scope('output'):
                # layer_output = [batch_size * seq_length, hidden_size]
                layer_output = tf.layers.dense(
                    inputs=intermediate_output,  # [batch_size * seq_length, intermediate_size]
                    units=hidden_size,
                    kernel_initializer=tf.truncated_normal_initializer(initializer_range))
                layer_output = tf.layers.dropout(
                    inputs=layer_output,
                    rate=hidden_dropout_prob)
                layer_output = tf.contrib.layers.layer_norm(
                    inputs=layer_output + attention_output,  # residual summation
                    begin_norm_axis=-1,  # default 1,
                    begin_params_axis=-1,  # default -1 이긴 하지만.
                    scope=None)
                prev_output = layer_output
                all_layer_outputs.append(prev_output)
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


class Model(object):
    def __init__(self, mode):
        '''

        :param mode: 0: search, 1: similarty
        '''
        self.mode = mode
        self.CONFIG = config.BERT
        self.preprocessor = PreProcessor()

        # placeholders
        self.input_ids = None
        self.input_masks = None
        self.segment_ids = None

        # pred indexes
        self.start_logits = None
        self.end_logtis = None
        self.start_pred = None
        self.end_pred = None

        # tf.Session()
        self.sess = None

        # feature vectors
        self.all_encoder_layers = None
        self.pooled_output = None
        self.feature_vector = None
        self.similarity_output = None

        self.build_model()

    def build_model(self):
        if self.mode == 0:
            bert_json = self.CONFIG['bert_json']
            max_seq_length = self.CONFIG['max_seq_length-search']
        elif self.mode == 1:
            bert_json = self.CONFIG['bert_json']
            model_path = self.CONFIG['model_path-similarity']
            max_seq_length = self.CONFIG['max_seq_length-similarity']

        bert_config = BertConfig()
        bert_config.read_from_json_file(bert_json)

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length])
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length])
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length])

        embedding_output = None  # sum of Token, segment, position
        embedding_table = None  # id embedding table
        self.all_encoder_layers = None  # transformer model
        self.similarity_output = None  # output layer
        self.elmo_output = None  # ELMO FEATURE 추출을 위한 레이어

        with tf.variable_scope(name_or_scope=None, default_name='bert'):
            with tf.variable_scope(name_or_scope='embeddings'):
                embedding_output, embedding_table = embedding_lookup(self.input_ids,
                                                                     bert_config.vocab_size,
                                                                     bert_config.hidden_size,
                                                                     bert_config.initializer_range,
                                                                     word_embedding_name='word_embeddings')
                embedding_output = embedding_postprocessor(embedding_output, use_token_type=True,
                                                           token_type_ids=self.segment_ids,
                                                           token_type_vocab_size=bert_config.type_vocab_size,
                                                           use_position_embeddings=True,
                                                           token_type_embedding_name='token_type_embeddings',
                                                           position_embedding_name='position_embeddings',
                                                           initializer_range=bert_config.initializer_range,
                                                           max_position_embeddings=bert_config.max_position_embeddings,
                                                           dropout_prob=bert_config.hidden_dropout_prob)

            with tf.variable_scope(name_or_scope='encoder'):
                attention_mask = create_attention_mask_from_input_mask(self.input_ids, self.input_masks)
                self.all_encoder_layers = tranformer_model(input_tensor=embedding_output,
                                                           attention_mask=attention_mask,
                                                           hidden_size=bert_config.hidden_size,
                                                           num_hidden_layers=bert_config.num_hidden_layers,
                                                           num_attention_heads=bert_config.num_attention_heads,
                                                           intermediate_size=bert_config.intermediate_size,
                                                           intermediate_act_fn=gelu,  # TODO gelu -> .
                                                           hidden_dropout_prob=bert_config.hidden_dropout_prob,
                                                           attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                                                           initializer_range=bert_config.initializer_range,
                                                           do_return_all_layers=True)

                self.similarity_output = self.all_encoder_layers[self.CONFIG['similarity_layer']]
                self.elmo_output = self.all_encoder_layers[-1]

            with tf.variable_scope('pooler'):
                first_token_tensor = tf.squeeze(self.similarity_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(inputs=first_token_tensor,
                                                     units=bert_config.hidden_size,
                                                     activation=tf.nn.tanh,
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         bert_config.initializer_range))

        final_layer = self.similarity_output

        output_weights = tf.get_variable('cls/squad/output_weights',
                                         shape=[2, bert_config.hidden_size],
                                         initializer=tf.truncated_normal_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable('cls/squad/output_bias',
                                      shape=[2],
                                      initializer=tf.truncated_normal_initializer(bert_config.hidden_size))

        final_layer = tf.reshape(final_layer, shape=[-1, bert_config.hidden_size])
        logits = tf.matmul(final_layer, output_weights, transpose_b=True) + output_bias

        logits = tf.reshape(logits, shape=[1, -1, 2])  # 질문이 하나씩 온다는 가정임
        logits = tf.transpose(logits, perm=[2, 0, 1])

        unstacked_logits = tf.unstack(logits, axis=0)

        self.start_logits = unstacked_logits[0]
        self.end_logtis = unstacked_logits[1]

        self.start_pred = tf.argmax(self.start_logits, axis=-1)
        self.end_pred = tf.argmax(self.end_logtis, axis=-1)

    def load_checkpoint(self):
        if self.mode == 0:
            model_path = self.CONFIG['model_path-search']
        elif self.mode == 1:
            model_path = self.CONFIG['model_path-similarity']

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, model_path)  # 201
        tf.train.init_from_checkpoint(model_path, assignment_map)
        self.sess = tf.Session()  # TODO 두번 불러야 정상작동되는 에러 해결
        self.sess.run(tf.global_variables_initializer())
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, model_path)  # 201
        tf.train.init_from_checkpoint(model_path, assignment_map)

        for var in tvars:
            if var.name in initialized_variable_names:
                print(var.name, ' - INIT FROM CKPT')

    def _convert_to_feature(self, chat, context):
        return self.preprocessor.create_InputFeature(chat, context=context)

    def predict(self, chat, text):

        input_feature = self._convert_to_feature(chat, text)

        feed_dict = {self.input_ids: np.array(input_feature.input_ids).reshape((1, -1)),
                     self.input_masks: np.array(input_feature.input_masks).reshape(1, -1),
                     self.segment_ids: np.array(input_feature.segment_ids).reshape(1, -1)}

        start, end = self.sess.run([self.start_pred, self.end_pred], feed_dict)
        # start_n, end_n = sess.run([start_n_best, end_n_best], feed_dict) # TODO n best answers

        return self.preprocessor.idx_to_orig(start, end, input_feature)

    def extract_feature_vector(self, input_feature):
        tic = time.time()
        length = np.sum(input_feature.input_masks)
        feed_dict = {self.input_ids: np.array(input_feature.input_ids).reshape((1, -1)),
                     self.input_masks: np.array(input_feature.input_masks).reshape(1, -1),
                     self.segment_ids: np.array(input_feature.segment_ids).reshape(1, -1)}
        sequence_output = self.sess.run(self.similarity_output, feed_dict)
        feature_vector = np.mean(sequence_output[:, 1:length - 1], axis=1)  # [CLS] 와 [SEP]를 제외한 단어 벡터들을 더함
        toc = time.time()
        print('*** Vectorizing Done: %5.3f ***' % (toc - tic))
        return np.reshape(feature_vector, newshape=(-1))

    # def extract_elmo_feature_vector(self, input_feature):
    #     tic = time.time()
    #     feed_dict = {self.input_ids: np.array(input_feature.input_ids).reshape((1, -1)),
    #                  self.input_masks: np.array(input_feature.input_masks).reshape(1, -1),
    #                  self.segment_ids: np.array(input_feature.segment_ids).reshape(1, -1)}
    #     elmo_output = self.sess.run(self.elmo_output, feed_dict)

    def search_to_saved_model(self):
        MODEL_DIR = self.CONFIG['MODEL_DIR']
        version = self.CONFIG['version-search']
        export_path = os.path.join(MODEL_DIR, 'search', str(version))
        print('export_path = {}\n'.format(export_path))
        if os.path.isdir(export_path):
            print('\nAlready saved a model, cleaning up\n')
            return
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        input_ids = tf.saved_model.utils.build_tensor_info(self.input_ids)
        input_masks = tf.saved_model.utils.build_tensor_info(self.input_masks)
        segment_ids = tf.saved_model.utils.build_tensor_info(self.segment_ids)

        start_pred = tf.saved_model.utils.build_tensor_info(self.start_logits)
        end_pred = tf.saved_model.utils.build_tensor_info(self.end_logtis)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_ids': input_ids,
                        'input_masks': input_masks,
                        'segment_ids': segment_ids},
                outputs={'start_pred': start_pred,
                         'end_pred': end_pred},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }

        builder.add_meta_graph_and_variables(self.sess,
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)

        builder.save()
        print('GENERATED SAVED MODEL')

    def ef_to_saved_model(self):
        MODEL_DIR = self.CONFIG['MODEL_DIR']
        version = self.CONFIG['version-similarity']
        export_path = os.path.join(MODEL_DIR, 'similarity', str(version))
        print('export_path = {}\n'.format(export_path))
        if os.path.isdir(export_path):
            print('\nAlready saved a model, cleaning up\n')
            return
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        input_ids = tf.saved_model.utils.build_tensor_info(self.input_ids)
        input_masks = tf.saved_model.utils.build_tensor_info(self.input_masks)
        segment_ids = tf.saved_model.utils.build_tensor_info(self.segment_ids)

        similarity_output = tf.saved_model.utils.build_tensor_info(self.similarity_output)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_ids': input_ids,
                        'input_masks': input_masks,
                        'segment_ids': segment_ids},
                outputs={'similarity_output': similarity_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }

        builder.add_meta_graph_and_variables(self.sess,
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)

        builder.save()
        print('GENERATED SAVED MODEL')

    # def load_saved_model(self, features):
    #
    #     input_ids = np.array(features.input_ids)
    #     input_masks = np.array(features.input_masks)
    #     segment_ids = np.array(features.segment_ids)
    #
    #     input_ids = np.reshape(input_ids, (-1, self.CONFIG['max_seq_length-search']))
    #     input_masks = np.reshape(input_masks, (-1, self.CONFIG['max_seq_length-search']))
    #     segment_ids = np.reshape(segment_ids, (-1, self.CONFIG['max_seq_length-search']))
    #
    #     sess = tf.Session()
    #     signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    #     placeholder_input_ids = 'input_ids'
    #     placeholder_input_masks = 'input_masks'
    #     placeholder_segment_ids = 'segment_ids'
    #     start_pred = 'start_pred'
    #     end_pred = 'end_pred'
    #     export_path = '/tmp/1/1'
    #
    #     meta_graph_def = tf.saved_model.loader.load(sess,
    #                                                 [tf.saved_model.tag_constants.SERVING],
    #                                                 export_path)
    #
    #     signature = meta_graph_def.signature_def
    #     input_ids_name = signature[signature_key].inputs[placeholder_input_ids].name
    #     input_masks_name = signature[signature_key].inputs[placeholder_input_masks].name
    #     segment_ids_name = signature[signature_key].inputs[placeholder_segment_ids].name
    #     start_pred_name = signature[signature_key].outputs[start_pred].name
    #     end_pred_name = signature[signature_key].outputs[end_pred].name
    #
    #     placeholder_input_ids = sess.graph.get_tensor_by_name(input_ids_name)
    #     placeholder_input_masks = sess.graph.get_tensor_by_name(input_masks_name)
    #     placeholder_segment_ids = sess.graph.get_tensor_by_name(segment_ids_name)
    #     start_pred = sess.graph.get_tensor_by_name(start_pred_name)
    #     end_pred = sess.graph.get_tensor_by_name(end_pred_name)
    #
    #     start, end = sess.run([start_pred, end_pred], {placeholder_input_ids: input_ids,
    #                                                    placeholder_input_masks: input_masks,
    #                                                    placeholder_segment_ids: segment_ids})
    #
    #     start = np.argmax(start, axis=-1)
    #     end = np.argmax(end, axis=-1)
    #
    #     return self.preprocessor.idx_to_orig(start, end, features)


if __name__ == "__main__":
    model = Model(mode=1)
