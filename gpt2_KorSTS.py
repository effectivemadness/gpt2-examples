import os
import tensorflow as tf
from transformers import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import re


import wget
import zipfile

wget.download('https://github.com/NLP-kr/tensorflow-ml-nlp-tf2/releases/download/v1.0/gpt_ckpt.zip')

with zipfile.ZipFile('gpt_ckpt.zip') as z:
    z.extractall()


# 시각화

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)


TOKENIZER_PATH = './gpt_ckpt/gpt2_kor_tokenizer.spiece'

tokenizer = SentencepieceTokenizer(TOKENIZER_PATH, alpha=0)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(TOKENIZER_PATH,
                                               mask_token=None,
                                               sep_token='<unused0>',
                                               cls_token=None,
                                               unknown_token='<unk>',
                                               padding_token='<pad>',
                                               bos_token='<s>',
                                               eos_token='</s>')


# KoSTS Simliarity 

tf.random.set_seed(0)
np.random.seed(0)

BATCH_SIZE = 10
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
SENT_MAX_LEN = 14

DATA_IN_PATH = 'data_in/KOR'
DATA_OUT_PATH = "data_out/KOR"

# Load Train dataset

TRAIN_STS_DF = os.path.join(DATA_IN_PATH, 'KorSTS', 'sts-train.tsv')
DEV_STS_DF = os.path.join(DATA_IN_PATH, 'KorSTS', 'sts-dev.tsv')

train_data = pd.read_csv(TRAIN_STS_DF, header=0, delimiter = '\t', quoting = 3)
dev_data = pd.read_csv(DEV_STS_DF, header=0, delimiter = '\t', quoting = 3)

train_data = train_data.dropna()

dev_data = dev_data.dropna()

print("Total # dataset: train - {}, dev - {}".format(len(train_data), len(dev_data)))


# 토크나이저를 제외하고는 5장에서 처리한 방식과 유사하게 접근
def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean

train_data_sents1 = []
train_data_sents2 = []
train_labels = []


for sent1, sent2, score in train_data[['sentence1', 'sentence2', 'score']].values:
    train_tokenized_sent_1 = vocab[tokenizer(clean_text(sent1))]
    train_tokenized_sent_2 = vocab[tokenizer(clean_text(sent2))]
    tokens1 = [vocab[vocab.bos_token]] 
    tokens1 += pad_sequences([train_tokenized_sent_1], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens1 += [vocab[vocab.sep_token]]  
    tokens1 += pad_sequences([train_tokenized_sent_2], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens1 += [vocab[vocab.eos_token]]
    tokens2 = [vocab[vocab.bos_token]] 
    tokens2 += pad_sequences([train_tokenized_sent_2], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens2 += [vocab[vocab.sep_token]]  
    tokens2 += pad_sequences([train_tokenized_sent_1], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens2 += [vocab[vocab.eos_token]]
    
    train_data_sents1.append(tokens1)
    train_data_sents2.append(tokens2)
    train_labels.append(score)

train_data_sents1 = np.array(train_data_sents1, dtype=np.int64)
train_data_sents2 = np.array(train_data_sents2, dtype=np.int64)
train_data_sents = (train_data_sents1, train_data_sents2)
train_data_labels = np.array(train_labels)


dev_data_sents1 = []
dev_data_sents2 = []
dev_labels = []


for sent1, sent2, score in dev_data[['sentence1', 'sentence2', 'score']].values:
    dev_tokenized_sent_1 = vocab[tokenizer(clean_text(sent1))]
    dev_tokenized_sent_2 = vocab[tokenizer(clean_text(sent2))]
    tokens1 = [vocab[vocab.bos_token]] 
    tokens1 += pad_sequences([dev_tokenized_sent_1], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens1 += [vocab[vocab.sep_token]]  
    tokens1 += pad_sequences([dev_tokenized_sent_2], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens1 += [vocab[vocab.eos_token]]
    tokens2 = [vocab[vocab.bos_token]] 
    tokens2 += pad_sequences([dev_tokenized_sent_2], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens2 += [vocab[vocab.sep_token]]  
    tokens2 += pad_sequences([dev_tokenized_sent_1], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens2 += [vocab[vocab.eos_token]]
    
    dev_data_sents1.append(tokens1)
    dev_data_sents2.append(tokens2)
    dev_labels.append(score)

dev_data_sents1 = np.array(dev_data_sents1, dtype=np.int64)
dev_data_sents2 = np.array(dev_data_sents2, dtype=np.int64)
dev_data_sents = (dev_data_sents1, dev_data_sents2)
dev_data_labels = np.array(dev_labels)


print("Shape of dataset: train - ({}, {}), dev - ({}, {})".format(train_data_sents[0].shape, train_data_sents[1].shape, dev_data_sents[0].shape, dev_data_sents[1].shape))


class TFGPT2Regressor(tf.keras.Model):
    def __init__(self, dir_path, num_class):
        super(TFGPT2Regressor, self).__init__()
        
        self.gpt2 = TFGPT2Model.from_pretrained(dir_path)
        self.num_class = num_class
        self.dropout = tf.keras.layers.Dropout(self.gpt2.config.summary_first_dropout)
        self.regressor = tf.keras.layers.Dense(self.num_class, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.gpt2.config.initializer_range), 
                                                name="regressior")
        
    def call(self, inputs):
        outputs1 = self.gpt2(inputs[0])
        outputs2 = self.gpt2(inputs[1])
        outputs = outputs1[0] + outputs2[0]
        pooled_output = outputs[:, -1, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)

        return logits


regression_model = TFGPT2Regressor('./gpt_ckpt', 1)


class PearsonCorrelationMetric(tf.keras.metrics.Metric):
    def __init__(self, name="pearson_correlation", **kwargs):
        super(PearsonCorrelationMetric, self).__init__(name=name, **kwargs)
        self.y_true_list = []
        self.y_pred_list = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=[-1])
        y_pred = tf.reshape(y_pred, shape=[-1])
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)

    def result(self):
        y_true = tf.concat(self.y_true_list, -1)
        y_pred = tf.concat(self.y_pred_list, -1)
        pearson_correlation = self.pearson(y_true, y_pred)
        
        return pearson_correlation

    def reset_states(self):
        self.y_true_list = []
        self.y_pred_list = []
        

    def pearson(self, true, pred):
        m_true = tf.reduce_mean(true)
        m_pred = tf.reduce_mean(pred)
        m_true, m_pred = true-m_true, pred-m_pred
        r_num = tf.reduce_sum(tf.multiply(m_true, m_pred))
        r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(m_true)), tf.reduce_sum(tf.square(m_pred)))) + 1e-12
        return r_num / r_den


optimizer = tf.keras.optimizers.Adam(6.25e-5)
loss = tf.keras.losses.MeanSquaredError()
metric = PearsonCorrelationMetric()
regression_model.compile(optimizer=optimizer, loss=loss, metrics=[metric], run_eagerly=True)


model_name = "tf2_gpt_korsts"

earlystop_callback = EarlyStopping(monitor='val_pearson_correlation', min_delta=0.0001,patience=3,mode='max')

checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_pearson_correlation', verbose=1, save_best_only=True, save_weights_only=True,mode='max')

history = regression_model.fit(train_data_sents, train_data_labels, epochs=NUM_EPOCHS,
                        validation_data = (dev_data_sents, dev_data_labels),
            batch_size=BATCH_SIZE, callbacks=[earlystop_callback, cp_callback])



plot_graphs(history, 'pearson_correlation')



plot_graphs(history, 'loss')


# KorSTSTEST

# Load Test dataset
TEST_STS_DF = os.path.join(DATA_IN_PATH, 'KorSTS', 'sts-test.tsv')

test_data = pd.read_csv(TEST_STS_DF, header=0, delimiter = '\t', quoting = 3)
test_data = test_data.dropna()
test_data.head()


test_data_sents1 = []
test_data_sents2 = []
test_labels = []


for sent1, sent2, score in test_data[['sentence1', 'sentence2', 'score']].values:
    test_tokenized_sent_1 = vocab[tokenizer(clean_text(sent1))]
    test_tokenized_sent_2 = vocab[tokenizer(clean_text(sent2))]
    tokens1 = [vocab[vocab.bos_token]] 
    tokens1 += pad_sequences([test_tokenized_sent_1], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens1 += [vocab[vocab.sep_token]]  
    tokens1 += pad_sequences([test_tokenized_sent_2], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens1 += [vocab[vocab.eos_token]]
    tokens2 = [vocab[vocab.bos_token]] 
    tokens2 += pad_sequences([test_tokenized_sent_2], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens2 += [vocab[vocab.sep_token]]  
    tokens2 += pad_sequences([test_tokenized_sent_1], 
                            SENT_MAX_LEN, 
                            value=vocab[vocab.padding_token], 
                            padding='post').tolist()[0] 
    tokens2 += [vocab[vocab.eos_token]]
    
    test_data_sents1.append(tokens1)
    test_data_sents2.append(tokens2)
    test_labels.append(score)

test_data_sents1 = np.array(test_data_sents1, dtype=np.int64)
test_data_sents2 = np.array(test_data_sents2, dtype=np.int64)
test_data_sents = (test_data_sents1, test_data_sents2)
test_data_labels = np.array(test_labels)


print("# sents: {}, # labels: {}".format(len(test_data_sents), len(test_data_labels)))


regression_model.load_weights(checkpoint_path)

results = regression_model.evaluate(test_data_sents, test_data_labels, batch_size=512)
print("test loss, test pearson correlation: ", results)
