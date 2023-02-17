from tensorflow import keras
import tensorflow as tf
from helpful_methods import generate_dataset, maneuver_dict, maneuvers, enable_mirroring
import numpy as np
from graph_plot import draw_maneuvers, draw_updated_maneuvers
from texttable import Texttable
from maneuver import Maneuver
import pandas
import time
import os


# Name of the directory
LOG_NAME = f'{int(time.time())}'

# The model that will be used
model = keras.models.load_model('best_model.h5')

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(tf.config.list_physical_devices('GPU'))


def test_KI(amount: int) -> None:
    """
    Testing the neural network using the evaluate method given by tensorflow
    
    :param amount: size of the testing data
    """
    # Generating a dataset of test values
    x_test, y_test = generate_dataset(amount)

    # Calculating the accuracy of the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Printing the values
    print('Test accuracy:', test_accuracy)
    print('Test loss:', test_loss)


def predict_all(amount: int) -> None:
    """
    The standard test consisting of a single prediction of every maneuver.
    
    :param amount: the testing amount for each maneuver
    """
    for index, maneuver in enumerate(maneuvers):
        for rand_man in maneuver.generate_maneuvers(amount, mirror=enable_mirroring[index]):
            predict_single(rand_man, draw_plot=False)


def predict_single(maneuver: Maneuver, draw_plot: bool = True, allow_print: bool = True) -> bool:
    """
    Gives a prediction of a single maneuver displaying the percentage the
    maneuver resembles a maneuver. When used make sure the first input has the expected
    dimensions of the model to prevent errors.
    
    :param maneuver: the maneuver that will be tested
    :param draw_plot: True if the maneuver should be displayed in a plot
    :param allow_print: True if the output should be printed
    :return: True if the maneuver was predicted correctly
    """
    # drawing the plot
    if draw_plot:
        draw_maneuvers([maneuver])

    x_test = np.array([maneuver.get_numpy_array()])
    prediction = model.predict(x_test)

    # reformating the values from prediction
    probabilities = [round(float(val) * 100, 4) for pred in prediction for val in pred]

    # name of the predicted maneuver
    pred_string = maneuver_dict[probabilities.index(max(probabilities))]

    if allow_print:
        print(f'\nErkennung von Manöver {maneuver.get_name()}')
        for i in range(len(probabilities)):
            print(f'{maneuver_dict[i]:19}: {probabilities[i]}%')

    if pred_string == maneuver.get_name():
        if allow_print:
            print(f'\nVorhersage: \033[92m{pred_string}\033[0m\n')
        return True
    else:
        if allow_print:
            print(f'\nVorhersage: \033[31m{pred_string}\033[0m\n')
        return False


def predict_partial_maneuver(maneuver: Maneuver, draw_plot: bool = True) -> int:
    """
    Predicting a partial maneuver starting by a full maneuver and 
    decreasing the size until the maneuver is no longer correctly recognized.
    Also prints the length to which it was still recognized.
    
    :param maneuver: the maneuver that will be tested
    :param draw_plot: True if the maneuver should be displayed in a plot
    """
    for length in range(300, 1, -1):
        m = maneuver.get_partial(length)
        prediction = predict_single(m, allow_print=False, draw_plot=False)

        if not prediction:
            print(
                f'Das Manöver wurde mit einer Länge von {min(length + 1, len(maneuver))} noch richtig erkannt ({((min(length + 1, len(maneuver)) / len(maneuver)) * 100):.2f}%).')

            if draw_plot:
                print('\nIn blau dargestellt ist der Teil des Manövers, der notwendig war, um das Manöver zu erkennen.')
                print('Das komplette Manöver wird daneben in orange als Referenz abgebildet.\n')
                draw_m = maneuver.get_partial(length + 1)
                draw_updated_maneuvers([draw_m, maneuver.move(0, 400, 0)])

            return length + 1
    return len(maneuver)


def predict_partial_amount(maneuver: Maneuver, amount: int, allow_print: bool = True, draw_plot: bool = True) -> float:
    """
    Uses predict_partial_maneuver() multiple times to determine an average value
    for the length the maneuver is still recognized.
    
    :param maneuver: the maneuver that should be tested
    :param amount: the testing amount for the maneuver
    :param allow_print: True if the result should be printed
    """
    scores = []
    for m in maneuver.generate_maneuvers(amount):
        score = predict_partial_maneuver(m, draw_plot=False)
        scores.append(score)

        if draw_plot:
            print('\nIn blau dargestellt ist der Teil des Manövers, der notwendig war, um das Manöver zu erkennen.')
            print('Das komplette Manöver wird daneben in orange als Referenz abgebildet.\n')
            draw_m = maneuver.get_partial(round(score))
            draw_updated_maneuvers([draw_m, maneuver.move(0, 400, 0)])

    average_score = sum(scores) / len(scores)
    if allow_print:
        print(
            f'Das Manöver {maneuver.get_name()} wird bei einer durchschnittlichen Länge von {average_score:.2f} erkannt.')
        print(f'Dies entspricht {average_score / len(maneuver) * 100:.2f}% der Gesamtlänge des Manövers.')
    return average_score


def predict_partial_all(amount_per_maneuver: int, draw_plot=True) -> None:
    """
    Uses predict_partial_amount() to evaluate all maneuvers
    
    :param amount_per_maneuver: the training amount for every maneuver
    """
    all_scores = []

    for m in maneuvers:
        score = predict_partial_amount(m, amount_per_maneuver, draw_plot=False, allow_print=False)
        all_scores.append(score)
        if draw_plot:
            print('\nIn blau dargestellt ist der Teil des Manövers, der notwendig war, um das Manöver zu erkennen.')
            print('Das komplette Manöver wird daneben in orange als Referenz abgebildet.\n')
            draw_m = m.get_partial(round(score))
            draw_updated_maneuvers([draw_m, m.move(0, 400, 0)])

    rows = [['Manöver', 'Länge', 'in %']]
    for i, score in enumerate(all_scores):
        rows.append([maneuver_dict[i], f'{min(score, 300):.2f}', f'{min(score, 300) / 300 * 100:.2}'])

    t = Texttable()
    t.add_rows(rows)
    print(t.draw())


def create_csv_recognition_rate(maneuver: Maneuver, amount: int) -> None:
    """
    Creates an excel file containing values for a graph to display the prediction values for each length.

    :param maneuver: the maneuver that the file will be crated for
    :param amount: the size of the sample to create the data
    """
    # check if mirroring the maneuver is forbidden or not
    maneuver_index = list(maneuver_dict.values()).index(maneuver.get_name())
    rand_man = maneuver.generate_maneuvers(amount=amount, mirror=enable_mirroring[maneuver_index])

    csv_data = []
    for length in range(300, 0, -1):
        min_value = 1.1
        max_value = -1
        average = 0

        for rand_m in rand_man:
            tmp_m = rand_m.get_partial(length)
            tmp_man_arr = np.array([tmp_m.get_numpy_array()])
            prob = model.predict(tmp_man_arr)[0][maneuver_index]
            print(prob)

            # setting data
            average += prob
            min_value = min(min_value, prob)
            max_value = max(max_value, prob)

        csv_data.insert(0, [average / amount, min_value, max_value])

    # initializing pandas dataframe
    dt = pandas.DataFrame(
        data=csv_data, 
        index=range(1, 301),
        columns=['Durchschnittswert', 'Minimalwert', 'Maximalwert']
    )
    
    # create the file structure
    try:            
        os.makedirs(f'csv_files/amount_{amount}/{LOG_NAME}')
    except:
        pass

    # saving to csv file
    dt.to_csv(f'csv_files/amount_{amount}/{LOG_NAME}/{maneuver.get_name()}.csv') # /amount_{amount}/{int(time.time())}


if __name__ == '__main__':
    # predict_partial_all(150, draw_plot=False)
    # predict_all(10)
    # predict_all(10)
    # test_KI(50)
    for m in maneuvers:
        create_csv_recognition_rate(m, 100)
