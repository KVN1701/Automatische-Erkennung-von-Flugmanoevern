from tensorflow import keras
from helpful_methods import parse_file, generate_dataset, maneuver_dict
import numpy as np
from graph_plot import draw_maneuvers
from maneuver import Maneuver


model = keras.models.load_model('best_model.h5')


def test_KI(amount):
    # Generating a dataset of test values
    x_test, y_test = generate_dataset(amount)

    # Calculating the accuracy of the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Printing the values
    print('Test accuracy:', test_accuracy)
    print('Test loss:', test_loss)
    
    
def predict_single(maneuver, draw_plot=True):
    if draw_plot:
        draw_maneuvers([maneuver])
    x_test = np.array([maneuver.get_numpy_array()])
    prediction = model.predict(x_test)
    probabilities = [round(float(val) * 100, 4) for pred in prediction for val in pred]
    print('\nWahrscheinlichkeiten für Manöver:')
    for i in range(len(probabilities)):
        print(f'{maneuver_dict[i]:17}: {probabilities[i]}%')
    
    
if __name__ == '__main__':
    test_m = parse_file('Abschwung').generate_maneuvers(1)
    predict_single(test_m[0], False)
    test_KI(50)

