import numpy as np
import pandas as pd
import codecs
from gensim.models import KeyedVectors
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import GlobalMaxPool1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from hazm import *
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import f1_score

model = keras.models.load_model("CNN_model")

#TODO: load tweets dataset
#TODO: embed tweets text
#TODO: predict sentimental class using the trained model
#TODO: visualize for me