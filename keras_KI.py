import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Normalization
from FileParser import parse_file
import matplotlib.pyplot as plt


"""
! Wieder selbes Problem: Die Arrays müssen gleich lang sein
"""

train_amount = 100
test_amount = 40

maneuvers = [
    parse_file("Looping_01"),
    parse_file("JoJo_01")
]

for maneuver in maneuvers:
    maneuver.set_to_new_length(500) # max([len(m) for m in maneuvers])
    

def generate_dataset(amount):
    """
    generates a dataset to train the Neural Network
    
    :param amount: the amount of training data
    """
    num_of_maneuvers = len(maneuvers)
    x_dataset = []
    tmp = []
    for m in maneuvers:
        tmp.extend(m.generate_maneuvers(amount))
        
    for m in tmp:
        x_dataset.append(m.get_numpy_array())
        
    x_dataset = np.array(x_dataset, dtype=float)
    
    y_dataset = []
    for i in range(num_of_maneuvers):
        y_dataset.extend([i for _ in range(amount)])
        
    y_dataset = np.array(y_dataset, dtype=int)
    
    return x_dataset, y_dataset
    
    
x_train, y_train = generate_dataset(train_amount)
x_test, y_test = generate_dataset(test_amount)


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))



num_classes = len(np.unique(y_train))


idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]



def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling2D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)


epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

"""
## Evaluate model on test data
"""

model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

"""
## Plot the model's training and validation loss
"""

metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()