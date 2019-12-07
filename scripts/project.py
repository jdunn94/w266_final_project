"""
Top-level code for running end-to-end model
"""
import os


from tensorflow.python.keras.utils import to_categorical
import pickle
import pandas as pd
from bert_embedding import BertEmbedding

from attention import SarcasmAttention
from convolution import SarcasmCNN
from utils import Preprocessor, Plotter

TRAINING_DATA_PATH = "/home/jimmy_dunn/w266/JDunn_Final/data"
TRAINING_FILES = ["sarcasm_train.pickle", "sarcasm_two_train.pickle"]
CSV_FILES = ["sentiment140_labeled_done.csv"]

# first read in the data
data = dict()
for file in TRAINING_FILES:
    with open(os.path.join(TRAINING_DATA_PATH, file), "rb") as f:
        data[file] = pickle.load(f, encoding='latin1')

for file in CSV_FILES:
    data[file] = pd.read_csv(os.path.join(TRAINING_DATA_PATH, file), encoding="ISO-8859-1", header=0, 
                             names=["id", "text", "labels"])

# second, preprocess the data for model ingestion
pp = Preprocessor(data, debug=False)
X_base_train, y_base_train, vocab_processor_base, X_base_test, y_base_test = pp.preprocess(datasource="s140")
X1_train, y1_train, vocab_processor1, X1_test, y1_test = pp.preprocess(datasource="scv1")
X2_train, y2_train, vocab_processor2, X2_test, y2_test = pp.preprocess(datasource="scv2")
X_val, y_val = pp.preprocess(datasource="s140", split=False)

# third, run the models
# baseline
cnn_model_base = SarcasmCNN(data=((X_base_train, y_base_train),
                             (X_base_test, y_base_test)), vocab_processor=vocab_processor_base)
print("Baseline Performance")
cnn_model_base.run()
Plotter(cnn_model_base.history).plot_history()

# Pretrained on SCv1
cnn_model1 = SarcasmCNN(data=((X1_train, y1_train),
                             (X1_test, y1_test)), vocab_processor=vocab_processor1)
print("Internal validation on SCv1")
cnn_model1.run()
Plotter(cnn_model1.history).plot_history()

print("Test validation on hand labeled S140")
cnn_model1.evaluate(X_val, y_val)

# Pretrained on SCv2
cnn_model2 = SarcasmCNN(data=((X2_train, y2_train),
                             (X2_test, y2_test)), vocab_processor=vocab_processor2)
print("Internal validation on SCv2")
cnn_model2.run()
Plotter(cnn_model2.history).plot_history()
print("Test validation on hand labeled S140")
cnn_model2.evaluate(X_val, y_val)