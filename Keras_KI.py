import numpy as np
from tensorflow import keras
from FileParser import parse_file
import matplotlib.pyplot as plt
from time import time
from math import floor


total_time = time()

train_amount = 100 # 0.76 bei 1000 und 3 Manövern bei 1500 keine Verbesserung 3000 --> 100%
test_amount = 50


maneuvers = [
    parse_file("Looping"),
    parse_file("LangsamerJoJo"),
    parse_file("SchnellerJoJo")
]
    

def generate_dataset(amount):
    """
    generates a dataset to train the Neural Network
    
    :param amount: the amount of training data
    """
    num_of_maneuvers = len(maneuvers)
    x_dataset = []
    tmp = []
    for i in range(len(maneuvers)): # generating the maneuvers to train the neural network
        m = maneuvers[i]
        tmp.extend(m.generate_maneuvers(amount, title=i+1))
        
      
    for m in tmp:
        x_dataset.append(m.get_numpy_array())
        
    x_dataset = np.array(x_dataset, dtype=float)
    
    y_dataset = []
    for i in range(num_of_maneuvers):
        y_dataset.extend([i for _ in range(amount)])
        
    y_dataset = np.array(y_dataset, dtype=int)
    
    return x_dataset, y_dataset
    
    

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


time_train = time()

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


# time finished
timestamp = time()

# print the time the script took
def format_time(time_in_seconds: float) -> str:
    hours = floor(time_in_seconds/3600)
    minutes = floor((time_in_seconds - hours * 3600)/60)
    if minutes > 0:
        if hours > 0:
            return f'{hours} hours {minutes} minutes and {time_in_seconds - minutes * 60 - hours * 3600:.2f} seconds'
        return f'{minutes} minutes and {time_in_seconds - minutes * 60:.2f} seconds'
    return f'{time_in_seconds:.2f} seconds'


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
