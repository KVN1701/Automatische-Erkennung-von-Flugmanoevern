import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Normalization
from FileParser import parse_file

"""
! Bei der Normalisierung von mehreren Manövern tritt ein Fehler auf, der mit astype
! verschwindet. Allerdings tut sich nun ein Numpy Fehler auf. Wahrscheinlich, weil
! die einzelnen Listen der Manöver unterschiedliche Längen haben. Ist es möglich die 
! fehlenden Stellen mit None aufzufüllen?

? Nach Recherche: Möglicherweise RaggedTensor
? in to_tensor für ragged tensor schauen füllt die übrigen Stellen mit einem Standardwert
"""

"""
* Was wurde behoben? Zunächst einmal war eine Umformung von numpy nicht möglich, da die Dimensionen nicht festgelegt waren
* Die Daten wurden als raggedTensor angelegt und dann in einen Tensor umgebaut
* Dafür sind leere Felder mit 0 aufgefüllt worden
* Eine Verfälschung des Ergebnisses ist möglich
"""

# Translation for the labels list
int_to_maneuver = {
    0: 'Looping',
    1: 'Jo-Jo'
}


# Amount of variations of a Maneuver to train the neural network
training_amount = 1


# Listing all training data
training_maneuvers = [
    parse_file("Looping_01"),
    parse_file("JoJo_01")
]

longest_statelist = max([len(elem) for elem in training_maneuvers])

training_maneuvers = [m.generate_maneuvers(training_amount) for m in training_maneuvers]
training_data = np.array([m.get_numpy_array() for m_list in training_maneuvers for m in m_list], dtype=object) # ? dtype=object possible fix


# generating labels list (for further info see dict above)
training_labels = [0 for _ in range(training_amount)] # Einfügen von Labels für Manöver "Looping"
training_labels.extend([1 for _ in range(training_amount)]) # Einfügen von Labels für Manöver "JoJo"
training_labels = np.array(training_labels)


# ? making the training_data a ragged Tensor
# The ragged Tensor is shaped to a normal Tensor. Therefore the rest of the values are filled with 0
# ! muss noch geprüft werden, ob dies das Ergebnis der KI verfälscht
training_data = tf.ragged.constant(training_data, dtype=tf.float32)
training_data = training_data.to_tensor(default_value=None, shape=(len(training_maneuvers) * training_amount, longest_statelist, 3))


# normalizing the data in training data
normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

training_data = normalizer(training_data)
print(training_data)


# * test prints
# print("var: %.4f" % np.var(training_data))
# print("mean: %.4f" % np.mean(training_data))
print(training_data.shape)


model = keras.Sequential([
    keras.layers.Dense(units=16, input_shape=training_data.shape, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=len(training_maneuvers), activation='softmax')
])

model.summary()

model.compile()

model.fit(x=training_data, y=training_labels)
