import functools
import numpy as np
import pandas as pd
import tensorflow as tf

# if you want to grab file from internet
# train_file_path = tf.keras.utils.get_file("traincta.csv", "url")

train_file_path = "cta/train_cta.csv"
test_file_path = "cta/test_cta.csv"

LABEL_COLUMN = 'rides'

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

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

# show_batch(raw_train_data)

# Center/normalize numeric data
def normalize_numeric_data(data, mean, std):
    return (data-mean)/std

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    if numeric_features != []:
      numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

# you can select specific columns if need be
SELECT_COLUMNS = ['stationname', 'date', 'daytype', 'rides']

raw_train_data = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
raw_test_data = get_dataset(test_file_path)

show_batch(raw_train_data)

# create a list of names for columns containing numeric values to be packed
NUMERIC_FEATURES = []
packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

# below shows a random batch of data from the packed dataset
show_batch(packed_train_data)

# create a batch of packed data
example_batch, labels_batch = next(iter(packed_train_data))

# get a description of the data from pandas to retrieve the mean and standard deviation
# The mean based normalization used here requires knowing the means of each column ahead of time
df = pd.read_csv(train_file_path)

nf = df[NUMERIC_FEATURES]
if not nf.empty:
  desc = nf.describe()
  MEAN = np.array(desc.T['mean'])
  STD = np.array(desc.T['std'])

  normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

  numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
  numeric_columns = [numeric_column]

  numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
  numeric_layer(example_batch).numpy()
else:
  numeric_columns = []

# define the possible values for each column category
CATEGORIES = {
    'stationname' : df['stationname'].drop_duplicates().tolist(),
    'date' : df['date'].drop_duplicates().tolist(),
    'daytype' : df['date'].drop_duplicates().tolist()
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

# below prints the first row of categorical data from the random example batch of packed training data
print(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])
print(preprocessing_layer(example_batch).numpy()[0])

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer='adam',
    metrics=['accuracy']
)

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

model.fit(train_data, epochs=10)

test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
print('')

predictions = model.predict(test_data)

# Show some results
for prediction, rides in zip(predictions[:10], list(test_data)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted rides: {:}".format(prediction[0]),
        " | Actual outcome: ", rides)
