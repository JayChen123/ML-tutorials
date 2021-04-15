from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from keras import Sequential
from keras import layers as L
from math import ceil
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import Recall
import numpy as np
from argparse import ArgumentParser
from keras.callbacks import Callback
from keras.callbacks import TensorBoard

arg_parser = ArgumentParser()
arg_parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size的值')
arg_parser.add_argument('-max_len', '--max_len', type=int, help='序列长度', default=500)
arg_parser.add_argument('-epoch', '--epoch', default=10, type=int, help='训练的轮数')
arg_parser.add_argument('-lr', '--lr', type=float, default=1e-4, help="学习率")
args = arg_parser.parse_args()

NUM_WORDS = 40000
word_index = imdb.get_word_index()

MAX_LEN = args.max_len
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
learning_rate = args.lr


class IMDBDataLoader(Sequence):
    def __init__(self, x, y, batch_size=32, max_len=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.max_len = max_len

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        return pad_sequences(batch_x, padding='post', truncating='pre', maxlen=self.max_len), batch_y


def build_cnn_model():
    cnn = Sequential([
        L.Embedding(len(word_index) + 1, 100, input_length=MAX_LEN),
        L.Conv1D(filters=128, kernel_size=7, strides=1, padding='same', activation='tanh'),
        L.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='tanh'),
        L.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='tanh'),
        L.GlobalMaxPool1D(),
        L.Dense(256, activation='relu'),
        L.Dense(1, activation='sigmoid')
    ])
    return cnn


class EvalCallback(Callback):
    def __init__(self):
        super(EvalCallback, self).__init__()
        self.acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] > self.acc:
            self.acc = logs['val_accuracy']
            self.model.save_weights('best-weighs.weights')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
    cnn = build_cnn_model()
    cnn.summary()
    cnn.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate), metrics=['accuracy', Recall()])
    cnn.fit(IMDBDataLoader(x_train, y_train, max_len=MAX_LEN),
            validation_data=IMDBDataLoader(x_test, y_test, max_len=MAX_LEN),
            epochs=EPOCH, callbacks=[EvalCallback(), TensorBoard()])
