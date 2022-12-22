import numpy as np
from tensorflow import keras
from HelpfulMethods import parse_file, format_time
import matplotlib.pyplot as plt
from time import time


"""
! Bei training amount von 1000 braucht die KI 36 minutes and 26.42 seconds und hat eine genauigkeit von 70%
! ohne reshaping nur 7 minutes and 2.22 seconds und 72%
! 
! 49 minutes and 2.52 seconds Test accuracy 98% amount 1500
! 44 minutes and 20.04 seconds 72% Layer 300 300 300 amount 500
! 11 minutes and 58.03 seconds 66.4% Layer 300 100 50 amount 500
! 9 minutes and 36.72 seconds 65% Layer 300 150 75 amount 500 23 minutes and 49.33 seconds 67,3% amount 1000
! 35 minutes and 36.25 seconds 65% Layer 500 250 100 amount 500

! 19 minutes and 28.86 seconds 76,7% Layer 128 96 64 amount 1000
! took 16 minutes and 57.06 seconds 73,3% Layer 128 96 64 amount 1500 98% 2 Stunden
"""


total_time = time()

# The amount of maneuvers that will be generated for every maneuver in maneuvers
train_amount = 5000 # 0.76 bei 1000 und 3 Manövern bei 1500 keine Verbesserung, 3000 --> 100%

# The amount of test maneuvers that will be generated
test_amount = 50


# one maneuver consists of 300 states
maneuvers = [
    parse_file("Looping"),
    parse_file("LangsamerJoJo"),
    parse_file("SchnellerJoJo")
]
    

def generate_dataset(amount):
    """
    Generates a dataset to train the Neural Network.
    
    :param amount: the amount of training data
    :return: the dataset in a tuple containing the data and labels
    """
    num_of_maneuvers = len(maneuvers)
    
    # generating the maneuvers to train/ test the neural network
    x_dataset = []
    tmp = []
    for i in range(len(maneuvers)): 
        m = maneuvers[i]
        tmp.extend(m.generate_maneuvers(amount, title=i+1))
        
    # converting the maneuvers to numpy arrays
    for m in tmp:
        x_dataset.append(m.get_numpy_array())   
    x_dataset = np.array(x_dataset, dtype=float)
    
    # generating the labels for the dataset
    y_dataset = []
    for i in range(num_of_maneuvers):
        y_dataset.extend([i for _ in range(amount)])
    y_dataset = np.array(y_dataset, dtype=int)
    
    return x_dataset, y_dataset
    
    
# Generating the datasets to train/ test the neural network
print('\nGenerating the training Data:')
x_train, y_train = generate_dataset(train_amount)

print('\nGenerating the test data:')
x_test, y_test = generate_dataset(test_amount)
print()


# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print(x_train.shape)

num_classes = len(np.unique(y_train))


idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]



def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer) # filter = Länge der Liste
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


time_train = time()

model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)


epochs = 500 # 500 nicht genug ohne reshape und 1500 train_amount
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


# time finished
timestamp = time()



# time to run the training
print('Training the neural network took ' + format_time(timestamp - time_train))

# time for the total script 
print('The total script took ' + format_time(timestamp - total_time))
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
