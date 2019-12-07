"""
Implementation of keras Attention Model
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate

BATCH_SIZE = 200
INDEX_OFFSET = 2
SEQUENCE_IDX = {
    "pad_id": 0,
    "start_id": 1,
    "oov_id": 2,
}
RNN_CELL_SIZE = 128
FORWARD_DROPOUT = 0.3
REVERSE_DROPOUT = 0.2
EPOCHS = 10
VALIDATION_SPLIT = 0.3


class AttentionModel(tf.keras.Model):
    """
    Implementation of simple Attention model

    Credit to: https://androidkt.com/text-classification-using-attention-mechanism-in-keras/
    """
    def __init__(self, units):
        """
        Constructor for AttentionModel following custom implementation of tf.keras.Model

        Params:
            units (int): positive integer, dimensionality of the output space
        """
        super().__init__()
        # first layer is Dense
        self.W_1 = tf.keras.layers.Dense(units)
        # second layer is Dense
        self.W_2 = tf.keras.layers.Dense(units)
        # final layer is single unit Dense corresponding to attention weight
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """Subclass override of tf.keras.Model method"""
        # add a dimension of 1 (time axis) in the second position in the hidden vector
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # pass a linear addition of dot product of W_1.dot(features) and W_2.dot(hidden_with_time_axis) into a tanh
        # activation
        score = tf.nn.tanh(self.W_1(features) + self.W_2(hidden_with_time_axis))
        # attention weights are the final layer passed into a softmax
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # output context_vector is the reduced sum of the attention_weights * features along columns
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights


class SarcasmAttention:
    """
    End-to-end attention pipeline for sarcasm classification on tweet corpus
    """
    def __init__(self, data, vocab):
        """
        Constructor for SarcasmAttention

        Params:
            data (nested tuple): train and test data pairs of form ((x_train, y_train), (x_test, y_test))
        """
        self.epochs = EPOCHS
        self.validation_split = VALIDATION_SPLIT
        self.batch_size = BATCH_SIZE
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.max_len = self.x_train.shape[1]
        self.rnn_cell_size = RNN_CELL_SIZE
        self.vocab = vocab
        self.vocab_size = len(self.vocab.vocabulary_)
        self.sequence_input = Input(shape=(self.max_len,), dtype='int32')
        self.embedded_sequences = keras.layers.Embedding(self.vocab_size, 128, input_length=self.max_len)(
            self.sequence_input)
        self.lstm, self.forward_h, self.forward_c, self.backward_h, self.backward_c = \
            self._init_bidirectional_rnn()
        self.state_h = Concatenate()([self.forward_h, self.backward_h])
        self.state_c = Concatenate()([self.forward_c, self.backward_c])
        self.model = self._init_model()
        self.early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                min_delta=0,
                                                                patience=1,
                                                                verbose=0, mode='auto')
        self.history = None  # keras.callbacks.History
        self.loss = None  # scalar test loss
        self.accuracy = None  # scalar test accuracy

    def _init_bidirectional_rnn(self):
        """Pair up a forward and a reverse LSTM to create a Bidrectional RNN"""
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                             (self.rnn_cell_size,
                                              dropout=0.3,
                                              return_sequences=True,
                                              return_state=True,
                                              recurrent_activation='relu',
                                              recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(
            self.embedded_sequences)

        return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                                               (self.rnn_cell_size,
                                                                dropout=0.2,
                                                                return_sequences=True,
                                                                return_state=True,
                                                                recurrent_activation='relu',
                                                                recurrent_initializer='glorot_uniform'))(lstm)

    def _init_model(self):
        context_vector, attention_weights = AttentionModel(self.rnn_cell_size)(self.lstm, self.state_h)
        output = keras.layers.Dense(1, activation='sigmoid')(context_vector)
        model = keras.Model(inputs=self.sequence_input, outputs=output)
        print(model.summary())
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def run(self):
        print(f"X_train shape: {self.x_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        self.history = self.model.fit(self.x_train,
                            self.y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_split=self.validation_split, verbose=1, callbacks=[self.early_stopping_callback])
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test)
        print("Testing Accuracy:  {:.4f}".format(self.accuracy))
        print(f"Testing Loss: {self.loss:.4f}")
