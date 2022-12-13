import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Normalization
from FileParser import parse_file
from sklearn.utils import shuffle
from Maneuver import Maneuver

"""
! Bei der Normalisierung von mehreren Manövern tritt ein Fehler auf, der mit astype
! verschwindet. Allerdings tut sich nun ein Numpy Fehler auf. Wahrscheinlich, weil
! die einzelnen Listen der Manöver unterschiedliche Längen haben. Ist es möglich die 
! fehlenden Stellen mit None aufzufüllen?
"""

# Translation for the labels list
int_to_maneuver = {
    0: 'Looping',
    1: 'Jo-Jo'
}


# Amount of variations of a Maneuver to train the neural network
training_amount = 50

training_maneuvers = [
    parse_file("Looping_01").generate_maneuvers(training_amount),
    # parse_file("JoJo_01").generate_maneuvers(training_amount)
]

training_data = np.array([m.get_numpy_array() for m_list in training_maneuvers for m in m_list]).astype('float64')

# generating labels list (for further info see dict above)
training_labels = [0 for _ in range(training_amount)]
training_labels.extend([1 for _ in range(training_amount)])
training_labels = np.array(training_labels)


# ? This line somehow causes an error in normalizer.adapt(training_data)
# training_labels, training_data = shuffle(training_labels, training_data)


# normalizing the data in training data
normalizer = Normalization(axis=1) # ! Error when adding a second Maneuver (Same Amount of points?)
normalizer.adapt(training_data)

training_data = normalizer(training_data)


# normalizing the training labels
# normalizer2 = keras.layers.Normalization(axis=-1)
# normalizer2.adapt(training_labels)

# training_labels = normalizer2(training_labels)

# * test prints
print("var: %.4f" % np.var(training_data))
print("mean: %.4f" % np.mean(training_data))
print(training_data.shape)


model = keras.Sequential([
    keras.layers.Dense(units=16, input_shape=training_data.shape, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=len(training_maneuvers), activation='softmax')
])

model.summary()
