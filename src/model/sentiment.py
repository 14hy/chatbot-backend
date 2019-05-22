import tensorflow as tf

# tf.enable_eager_execution()
import config
from src.model.saver import Model
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

"""
감정분석 학습 이후 모델 저장
2019.05.20
"""
CONFIG = config.BERT
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SENTIMENT_DATA_train = '/home/rhodochrosited/sentiment/nsmc/ratings_train.txt'
SENTIMENT_DATA_test = '/home/rhodochrosited/sentiment/nsmc/ratings_test.txt'
TFRECORD_DIR = '/home/rhodochrosited/sentiment/input_features'
TRAIN_FILE = 'train.tfrecord'
TEST_FILE = 'test.tfrecord'
TENSORBOARD_FILE_train = './board/train'
TENSORBOARD_FILE_test = './board/test'

MAX_SEQ_LENGTH = 25
BATCH_SIZE = 32
do_train = True
do_test = False
n_output = 1
learning_rate = 5e-5
STEPS = 15000
display_step = 10


def create_data(file, file_tf):
    # #  데이터 불러오기 및 저장
    #  데이터 불러오기
    DATA_train = pd.read_csv(file, sep='\t')
    print('데이터 크기: ', len(DATA_train))
    if os.path.exists(file_tf):
        print('FILE ALREADY EXISTS {}'.format(file_tf))
        return

    #  결측값 제거
    DATA_train.dropna(axis=0, inplace=True)
    #  문장, 라벨 추출
    X = DATA_train['document'].values
    Y = DATA_train['label'].values
    #  문장 전처리 및 토큰화
    from src.data.preprocessor import PreProcessor
    prep = PreProcessor()

    ##  전처리 1. 클린징
    X = list(map(lambda x: prep.clean(x)[0], X))

    ##  전처리 2. 토큰화 - InputFeatures object
    X = list(map(lambda x: prep.create_InputFeature(x), tqdm(X, desc='create_InputFeature')))

    #  write TFRecord dataset
    with tf.python_io.TFRecordWriter(file_tf) as writer:
        def _int64List_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        for i in tqdm(range(len(X)), desc='Writing to {}'.format(file_tf)):
            feature = {
                'input_ids': _int64List_feature(X[i].input_ids),
                'segment_ids': _int64List_feature(X[i].segment_ids),
                'input_masks': _int64List_feature(X[i].input_masks),
                'label': _int64_feature(Y[i])
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def create_model(pooler_output):
    output = tf.layers.Dense(units=n_output)
    output = output(pooler_output)

    return output


# Model 구성
#  GRU/ .. RNN모델? 보다. logistic regression

#  pooler output 을 사용해서 저장
#  모델 주기적 저장
#  모델 saved model로 저장.

if __name__ == '__main__':

    train_path = os.path.join(TFRECORD_DIR, TRAIN_FILE)
    test_path = os.path.join(TFRECORD_DIR, TEST_FILE)

    create_data(file=SENTIMENT_DATA_train, file_tf=train_path)
    create_data(file=SENTIMENT_DATA_test, file_tf=test_path)


    #  데이터 읽어오기
    def parse_fn(record):

        feature = {  # 데이터 스키마
            'input_ids': tf.FixedLenFeature(shape=[MAX_SEQ_LENGTH], dtype=tf.int64),
            'segment_ids': tf.FixedLenFeature(shape=[MAX_SEQ_LENGTH], dtype=tf.int64),
            'input_masks': tf.FixedLenFeature(shape=[MAX_SEQ_LENGTH], dtype=tf.int64),
            'label': tf.FixedLenFeature(shape=[1], dtype=tf.int64)
        }
        features = tf.parse_single_example(record, feature)

        i = tf.cast(features['input_ids'], dtype=tf.int32)
        segment = tf.cast(features['segment_ids'], dtype=tf.int32)
        masks = tf.cast(features['input_masks'], dtype=tf.int32)
        l = tf.cast(features['label'], dtype=tf.float32)
        return i, segment, masks, l


    if do_train:
        #  데이터 준비
        Dataset_train = tf.data.TFRecordDataset(filenames=train_path)
        Dataset_train = Dataset_train.map(parse_fn)
        Dataset_test = tf.data.TFRecordDataset(filenames=test_path)
        Dataset_test = Dataset_test.map(parse_fn)

        Dataset_train = Dataset_train.repeat().batch(BATCH_SIZE).shuffle(buffer_size=100)
        Dataset_test = Dataset_train.repeat().batch(BATCH_SIZE)
        iterator = Dataset_train.make_one_shot_iterator()
        iterator_test = Dataset_train.make_one_shot_iterator()
        input_ids, segment_ids, input_masks, labels = iterator.get_next()
        input_ids_test, segment_ids_test, input_masks_test, labels_test = iterator_test.get_next()

        model = Model(mode=1)  # similarity 버전으로 load

        input_label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input_label')

        logits = create_model(model.pooled_output)
        prob = tf.nn.sigmoid(logits, name='prob')

        predicts = tf.round(prob, name='predicts')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, input_label), dtype=tf.float32))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_label, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        tf.summary.scalar(name='loss', tensor=loss)
        tf.summary.scalar(name='accuracy', tensor=accuracy)
        merged = tf.summary.merge_all()

        model.load_checkpoint()
        last_test_loss = 100

        writer_train = tf.summary.FileWriter(logdir=TENSORBOARD_FILE_train, graph=model.sess.graph)
        writer_test = tf.summary.FileWriter(logdir=TENSORBOARD_FILE_test, graph=model.sess.graph)

        for step in range(STEPS):
            ii, si, im, l = model.sess.run([input_ids, segment_ids, input_masks, labels])

            feed_dict = {
                model.input_ids: ii,
                model.segment_ids: si,
                model.input_masks: im,
                input_label: l
            }
            _ = model.sess.run(optimizer, feed_dict=feed_dict)
            if step % display_step == 0:
                _, dl, da, summary = model.sess.run([optimizer, loss, accuracy, merged], feed_dict=feed_dict)
                writer_train.add_summary(summary, step)
                print('Train loss: {}, accruacy: {}'.format(dl, da))
                ii, si, im, l = model.sess.run([input_ids_test, segment_ids_test, input_masks_test, labels_test])

                feed_dict = {
                    model.input_ids: ii,
                    model.segment_ids: si,
                    model.input_masks: im,
                    input_label: l
                }
                _, dl, da, summary = model.sess.run([optimizer, loss, accuracy, merged], feed_dict=feed_dict)
                writer_test.add_summary(summary, step)
                print('테스트 로스: {}, 정확도: {}'.format(dl, da))

        print('TRAINING DONE')

        MODEL_DIR = CONFIG['MODEL_DIR']
        version = CONFIG['version-sentiment']
        export_path = os.path.join(MODEL_DIR, 'sentiment', str(version))
        print('export_path = {}\n'.format(export_path))
        # if os.path.isdir(export_path):
        #     print('\nAlready saved a model, cleaning up\n')

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        input_ids = tf.saved_model.utils.build_tensor_info(model.input_ids)
        input_masks = tf.saved_model.utils.build_tensor_info(model.input_masks)
        segment_ids = tf.saved_model.utils.build_tensor_info(model.segment_ids)

        predict = tf.saved_model.utils.build_tensor_info(prob)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_ids': input_ids,
                        'input_masks': input_masks,
                        'segment_ids': segment_ids},
                outputs={'predict': predict},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }

        builder.add_meta_graph_and_variables(model.sess,
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)

        builder.save()
        print('GENERATED SAVED MODEL')
