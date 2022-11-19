# Let's feel the society!

import numpy as np
import pandas as pd
import random
from googletrans import Translator
from google.colab import files
import time
from tqdm import tqdm
from hazm import *
import math

translator = Translator()
uploaded = files.upload()

x_train = pd.Series.from_csv('x_train.csv', sep='\t')
x_test = pd.Series.from_csv('x_test.csv', sep='\t')
y_train = pd.Series.from_csv('y_train.csv', sep='\t', header=0)
y_test = pd.Series.from_csv('y_test.csv', sep='\t', header=0)

x_train = x_train.iloc[1:, ]
x_test = x_test.iloc[1:, ]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

def translateAugmentation(sentence):
  try:
    # Translate sentence to English
    translate_to_en = translator.translate(sentence, src='fa', dest='en')
    # Translate back to Persian
    translate_to_fa = translator.translate(translate_to_en.text, src='en', dest='fa')
    # Return translated sentence
    return translate_to_fa.text
  except:
    return sentence

new_data = []
for sentence in tqdm(x_train):
  # print (sentence)
  translated = translateAugmentation(sentence)
  new_data.append(translated)
  # print(translated)

def find_similar(word):
  synonym = []
  try:
    translate_to_en = translator.translate(word, src='fa', dest='en')
    similar_words = translate_to_en.extra_data['all-translations'][0][2]
    for word in similar_words:
      for pword in word[1]:
        synonym.append(pword)
  except:
    synonym = [word]
  # Select a random synonym
  synonym = list(set(synonym))
  random_select = random.choice(synonym)
  return random_select

new_data = []
puncs = ['،', '.', ',', ':', ';', '"', '\\', '/', '*']
normalizer = Normalizer()
for sentence in tqdm(x_train):
  # print (sentence)
  doc = normalizer.normalize(sentence) # Normalize document using Hazm Normalizer
  tokenized = word_tokenize(doc)  # Tokenize text
  tokens = []
  for t in tokenized:
    temp = t
    for p in puncs:
      temp = temp.replace(p, '')
    tokens.append(temp)
  tokens = [w for w in tokens if not len(w) <= 1]
  tokens = [w for w in tokens if not w.isdigit()]
  # Change 20% of words with their synonym
  count = math.floor(((20 * len(tokens))/100))
  while count > 0 :
    # Make a random index
    random_index = random.randint(0,len(tokens)-1)
    # Get the word of index position
    random_word = tokens[random_index]
    # Find a similar word using translate
    translated = find_similar(random_word)
    # Replace with synonym word
    tokens[random_index] = translated
    count = count - 1
  result = ' '.join(tokens)
  new_data.append(result)
  # print(translated)

# Convert list to numpy array
new_data = np.asarray(new_data)

# Merge origin data and augmented data
final_x_train = np.append(x_train, new_data)
final_y_train = np.append(y_train, y_train)

x_train_df = pd.DataFrame(final_x_train)
y_train_df = pd.DataFrame(final_y_train)
x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

x_train_df.to_csv('x_train.csv', sep="\t")
y_train_df.to_csv('y_train.csv', sep="\t")
x_test_df.to_csv('x_test.csv', sep="\t")
y_test_df.to_csv('y_test.csv', sep="\t")

# files.download('x_train.csv')
# files.download('y_train.csv')
# files.download('x_test.csv')
# files.download('y_test.csv')


