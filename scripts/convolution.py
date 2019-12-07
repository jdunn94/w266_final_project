"""
Implementation of keras CNN architecture
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers


np.random.seed(42)

EPOCHS = 10
BATCH_SIZE = 5


class SarcasmCNN:
    """
    End-to-end convolutional neural network pipeline for sarcasm classification on tweet corpus
    """
    def __init__(self, data, vocab_processor):
        """
        Constructor for SarcasmCNN

        Params:
            data
        """
        # instantiate model
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        self.vocab_processor = vocab_processor
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.history = None
        self.loss = None
        self.accuracy = None
        self.model = Sequential()
        self._add_model_layers()

    def _add_model_layers(self):
        self.model.add(layers.Embedding(input_dim=len(self.vocab_processor.vocabulary_), output_dim=self.X_train.shape[1]))
        self.model.add(layers.Conv1D(128, 5, activation='relu'))
        self.model.add(layers.GlobalMaxPooling1D())
        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

    def fit(self):
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
        self.history = self.model.fit(self.X_train, self.y_train,
                                      epochs=self.epochs,
                                      verbose=True,
                                      validation_data=(self.X_test, self.y_test),
                                      batch_size=self.batch_size)

    def evaluate(self, X, y):
        loss, accuracy = self.model.evaluate(X, y, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        print(f"Testing Loss: {loss:.4f}")

    def run(self):
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
        self.history = self.model.fit(self.X_train, self.y_train,
                            epochs=self.epochs,
                            verbose=True,
                            validation_data=(self.X_test, self.y_test),
                            batch_size=self.batch_size)
        self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(self.accuracy))
        print(f"Testing Loss: {self.loss:.4f}")
