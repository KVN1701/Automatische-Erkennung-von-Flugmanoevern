import numpy as np
import tensorflow as tf
from tensorflow import keras
from helpful_methods import format_time, generate_dataset, maneuvers
import matplotlib.pyplot as plt
from time import time


"""
! 3000_Einheiten_4_Maneuver_weniger_Einheiten_oder_Manoever_reichen_auch.png 1 hours 55 minutes and 11.32 seconds
! 2000_Einheiten_4Manoever 42 min 57.60 sec
"""


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


total_time = time()

# The model that will be used
model = keras.models.load_model("tuner_models/1673558607.h5")

# The amount of maneuvers that will be generated for every maneuver in maneuvers
train_amount = 2000 # 0.76 bei 1000 und 3 Manövern bei 1500 keine Verbesserung, 3000 --> 100%

# The amount of test maneuvers that will be generated
test_amount = 50

    
# Generating the datasets to train/ test the neural network
print('\nGenerating the training Data:')
x_train, y_train = generate_dataset(train_amount)

print('\nGenerating the test data:')
x_test, y_test = generate_dataset(test_amount)
print()


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))


num_classes = len(np.unique(y_train))


idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


time_train = time()
keras.utils.plot_model(model, show_shapes=True)


callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]


history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=500,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

"""
## Evaluate model on test data
"""

model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

# printing the accuracy of the model
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)


# time finished
timestamp = time()


# time to run the training
print('Training time:', format_time(timestamp - time_train))

# time for the total script 
print('Script runtime:', format_time(timestamp - total_time))
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
fig = plt.gcf()
plt.show()
fig.savefig(f'D:/LOKALE DATEN/Programming/PycharmProjects/Bachelorarbeit.graphen/{train_amount}_Einheiten_{len(maneuvers)}_Manoever.png')
plt.close()
