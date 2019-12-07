"""
Utility module for preprocessing strings and plotting model performance
"""
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
TEST_SIZE = 0.2
DEBUG_SIZE = 10


class Preprocessor:
    """
    Transforms tweet-label vectors and tokenizes text
    """
    def __init__(self, data, debug=False):
        """
        Constructor for Preprocessor class

        Params:
            data (dict of str:np.array): complete data set read in from filesystem
        """
        self.data = data
        self.test_size = TEST_SIZE
        self.debug = debug
        self.s140 = self.data["sentiment140_labeled_done.csv"]
        self.scv1, self.scv2 = self._extract_deepmoji_data()

    def _extract_deepmoji_data(self):
        scv1_text = self.data["sarcasm_train.pickle"]["texts"]
        scv1_labels = [entry["label"] for entry in self.data["sarcasm_train.pickle"]["info"]]
        scv1_text_and_labels = {"text": scv1_text, "labels": scv1_labels}
        scv1 = pd.DataFrame(scv1_text_and_labels)

        scv2_text = self.data["sarcasm_two_train.pickle"]["texts"]
        scv2_labels = [entry["label"] for entry in self.data["sarcasm_two_train.pickle"]["info"]]
        scv2_text_and_labels = {"text": scv2_text, "labels": scv2_labels}
        scv2 = pd.DataFrame(scv2_text_and_labels)

        if self.debug:
            scv1 = scv1.sample(DEBUG_SIZE)
            scv2 = scv2.sample(DEBUG_SIZE)
        return scv1, scv2

    @staticmethod
    def clean_text(text):
        """
        Tokenization/text cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip().lower()

    def preprocess(self, datasource="scv1", split=True):
        # Helper function for cleaning text - strip emojis, very basic tokenization
        # start preprocessing
        print("Preparing data...")
        # clean text and extract labels
        if datasource == "scv1":
            dataframe = self.scv1
        elif datasource == "scv2":
            dataframe = self.scv2
        elif datasource == "s140":
            dataframe = self.s140
        else:
            return False

        x_text = dataframe.text.apply(self.clean_text).values
        y = dataframe.labels.values

        # build vocabulary, padding sentences to maximum document length
        max_document_length = max([len(text.split(" ")) for text in x_text])
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
        X = np.array(list(vocab_processor.fit_transform(x_text)))

        # split training and test sets
        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
            print(f"Vocabulary size: {len(vocab_processor.vocabulary_)}")
            print(f"Train/Test split: {len(y_train), len(y_test)}")
            return X_train, y_train, vocab_processor, X_test, y_test
        else:
            return X, y



class Plotter:
    """
    Helper class to plot model performance
    """
    def __init__(self, history):
        self.history = history

    def plot_history(self,):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
