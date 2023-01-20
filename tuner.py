import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Input, GlobalAveragePooling2D, ReLU, BatchNormalization, Flatten
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from helpful_methods import generate_dataset, format_time
import numpy as np
import time


# ? https://www.tensorflow.org/install/pip?_gl=1*1fjkigb*_ga*MTM1NzAzMjA1NS4xNjY3NDg2MzYw*_ga_W0YLR4190T*MTY3MzM4MjMzNC4xNC4wLjE2NzMzODIzMzQuMC4wLjA.#windows-wsl2_1
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(tf.config.list_physical_devices('GPU'))


"""
? ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
? tensorflow-gpu 2.10.1 requires keras<2.11,>=2.10.0, but you have keras 2.11.0 which is incompatible.
? tensorflow-gpu 2.10.1 requires tensorboard<2.11,>=2.10, but you have tensorboard 2.11.2 which is incompatible.
? tensorflow-gpu 2.10.1 requires tensorflow-estimator<2.11,>=2.10.0, but you have tensorflow-estimator 2.11.0 which is incompatible.

? ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
? tensorflow-intel 2.11.0 requires keras<2.12,>=2.11.0, but you have keras 2.10.0 which is incompatible.
? tensorflow-intel 2.11.0 requires tensorboard<2.12,>=2.11, but you have tensorboard 2.10.1 which is incompatible.
? tensorflow-intel 2.11.0 requires tensorflow-estimator<2.12,>=2.11.0, but you have tensorflow-estimator 2.10.0 which is incompatible.
"""


LOG_NAME = f"{int(time.time())}"
start_time = time.time()


tuner_train_amount = 100
tuner_test_amount = 50


x_train, y_train = generate_dataset(tuner_train_amount)
x_test, y_test = generate_dataset(tuner_test_amount)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

num_classes = len(np.unique(y_train))


idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


print(x_train.shape)
def build_model(hp):
    model = keras.models.Sequential()
    
    # creating an input layer
    model.add(Input(x_train.shape[1:]))
    
    # random amout of layers between 1 and 4
    for i in range(hp.Int("layers", 1, 4)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=512, step=32), kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())
        
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation="softmax"))
    
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    
    return model


tuner = RandomSearch(
    build_model,
    objective="sparse_categorical_accuracy",
    max_trials=100,
    executions_per_trial=2,
    directory=f"tuner_log/{LOG_NAME}"
)


early_stop = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1)
]


tuner.search(x=x_train,
             y=y_train,
             epochs=200,
             batch_size=32,
             validation_data=(x_test, y_test),
             callbacks=early_stop)
        

best_model = tuner.get_best_models()[0]
print(best_model.summary())


best_model.save(f"tuner_models/{LOG_NAME}.h5")
keras.utils.plot_model(best_model, show_shapes=True, to_file=f'tuner_models/{LOG_NAME}.png')


print(f"\nThe Script took {format_time(time.time() - start_time)}")