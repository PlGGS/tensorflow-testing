import functools
import numpy as np
import pandas as pd
import tensorflow as tf

train_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
test_file_path = tf.keras.utils.get_file("eval.csv", "https://storage.googleapis.com/tf-datasets/titanic/eval.csv")

LABEL_COLUMN = 'survived'
LABELS = [0, 1]

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5, # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True, 
        **kwargs)
    return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

# Center/normalize numeric data
def normalize_numeric_data(data, mean, std):
    return (data-mean)/std

# you can select specific columns if need be
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
# If your data is already in an appropriate numeric format, you can pack the data into a vector before passing it off to the model
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]

raw_train_data = get_dataset(train_file_path, select_columns=SELECT_COLUMNS, column_defaults = DEFAULTS)
raw_test_data = get_dataset(test_file_path)

# if you don't have mixed datatypes, you can pack like this
# def pack(features, label):
#     return tf.stack(list(features.values()), axis=-1), label
# packed_dataset = raw_train_data.map(pack)
# for features, labels in packed_dataset.take(1):
#     print(features.numpy())
#     print()
#     print(labels.numpy())

# create a list of names for columns containing numeric values to be packed
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

# create a batch of packed data
example_batch, labels_batch = next(iter(packed_train_data))

# below shows a random batch of data from the packed dataset
# show_batch(packed_train_data)

# get a description of the data from pandas to retrieve the mean and standard deviation
# The mean based normalization used here requires knowing the means of each column ahead of time
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()

# define the possible values for each column category
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

# below prints the first row of categorical data from the random example batch of packed training data
# print(categorical_layer(example_batch).numpy()[0])

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=20)

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels
