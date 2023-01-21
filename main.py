from tensorflow import keras
from helpful_methods import parse_file, generate_dataset, maneuver_dict
import numpy as np
from graph_plot import draw_maneuvers
from maneuver import *


model = keras.models.load_model('best_model.h5')


"""
! Immelmann rechts   : 43.6482%
! Immelmann links    : 56.3478%
! Erkennung korrekt aber Werte nicht nah genug aneinander
! 
! Immelmann rechts   : 16.6042%
! Immelmann links    : 83.3843%
! Werte durch verbieten von mirror deutlich verbessert, allerdings immer noch nicht optimal
"""


def test_KI(amount):
    # Generating a dataset of test values
    x_test, y_test = generate_dataset(amount)

    # Calculating the accuracy of the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Printing the values
    print('Test accuracy:', test_accuracy)
    print('Test loss:', test_loss)
    
    
def predict_single(maneuver, draw_plot=True):
    # drawing the plot
    if draw_plot:
        draw_maneuvers([maneuver])
        
    x_test = np.array([maneuver.get_numpy_array()])
    prediction = model.predict(x_test)
    
    # reformating the values from prediction
    probabilities = [round(float(val) * 100, 4) for pred in prediction for val in pred]
    pred_string = maneuver_dict[probabilities.index(max(probabilities))]
    
    print(f'\nErkennung von Manöver {maneuver.get_name()}')
    
    for i in range(len(probabilities)):
        print(f'{maneuver_dict[i]:19}: {probabilities[i]}%')
        
    if pred_string == maneuver.get_name():
        print(f'\nVorhersage: \033[92m{pred_string}\033[0m')
    else:
        print(f'\nVorhersage: \033[31m{pred_string}\033[0m')
    
    
 
if __name__ == '__main__':
    amount = 40
    singular_amount = round(amount/len(maneuver_dict))
    
    test_m = [
        parse_file('Abschwung').generate_maneuvers(singular_amount),
        parse_file('LangsamerJoJo').generate_maneuvers(singular_amount),
        parse_file('Looping').generate_maneuvers(singular_amount),
        parse_file('SchnellerJoJo').generate_maneuvers(singular_amount),
        parse_file('Kertwende').generate_maneuvers(singular_amount),
        parse_file('Immelmann_rechts').generate_maneuvers(singular_amount, mirror=False),
        parse_file('Immelmann_links').generate_maneuvers(singular_amount, mirror=False)
    ]
    
    for sublist in test_m:
        for m in sublist:
            predict_single(m, False)
    test_KI(50)

