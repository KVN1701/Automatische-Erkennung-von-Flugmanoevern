from tensorflow import keras
from HelpfulMethods import parse_file, generate_dataset, maneuver_dict
import numpy as np
from GraphPlot import draw_maneuvers
from Maneuver import Maneuver


model = keras.models.load_model('best_model.h5')


def test_KI(amount):
    # Generating a dataset of test values
    x_test, y_test = generate_dataset(amount)

    # Calculating the accuracy of the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Printing the values
    print('Test accuracy:', test_accuracy)
    print('Test loss:', test_loss)
    
    
def predict_single(maneuver):
    x_test = np.array([maneuver.get_numpy_array()])
    prediction = model.predict(x_test)
    probabilities = [float(f'{val:.5f}') for pred in prediction for val in pred]
    print('\nWahrscheinlichkeiten für Manöver:')
    for i in range(len(probabilities)):
        print(f'{maneuver_dict[i]:17}: {probabilities[i] * 100}%')
    
    
if __name__ == '__main__':
    test_m = parse_file('Looping').generate_maneuvers(1)
    draw_maneuvers(test_m)
    predict_single(test_m[0])

