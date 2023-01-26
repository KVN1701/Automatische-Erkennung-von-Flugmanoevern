from tensorflow import keras
from helpful_methods import parse_file, generate_dataset, maneuver_dict, maneuvers
import numpy as np
from graph_plot import *
from maneuver import *


model = keras.models.load_model('best_model.h5')


"""+
* Das Manöver Looping wird bei einer durchschnittlichen Länge von 232.68 erkannt.
* Dies entspricht 77.56% der Gesamtlänge des Manövers.
* 
* Das Manöver LangsamerJoJo wird bei einer durchschnittlichen Länge von 200.08 erkannt.
* Dies entspricht 66.69% der Gesamtlänge des Manövers.
* 
* Das Manöver SchnellerJoJo wird bei einer durchschnittlichen Länge von 206.13 erkannt.
* Dies entspricht 68.71% der Gesamtlänge des Manövers.
* 
* Das Manöver Abschwung wird bei einer durchschnittlichen Länge von 163.42 erkannt.
* Dies entspricht 54.47% der Gesamtlänge des Manövers.
* 
* Das Manöver Kertwende wird bei einer durchschnittlichen Länge von 255.28 erkannt.
* Dies entspricht 85.09% der Gesamtlänge des Manövers.
* 
* Das Manöver Immelmann_rechts wird bei einer durchschnittlichen Länge von 242.00 erkannt.
* Dies entspricht 80.67% der Gesamtlänge des Manövers.
* 
* Das Manöver Immelmann_links wird bei einer durchschnittlichen Länge von 232.66 erkannt.
* Dies entspricht 77.55% der Gesamtlänge des Manövers.
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
    print(x_test.shape)
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
    
    
    
def standard_test(amount):
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
    
    

def pred_partial_single(maneuver, allow_print=True):
    partial_m_arr = np.array([maneuver.get_numpy_array()])
    
    # ? Warum muss vorher eine prediction eines vollen Manövers ausgeführt werden?
    #model.predict(np.array([maneuver.get_numpy_array()]))
    prediction = model.predict(partial_m_arr)
    
    # reformating the values from prediction
    probabilities = [round(float(val) * 100, 4) for pred in prediction for val in pred]
    pred_string = maneuver_dict[probabilities.index(max(probabilities))]
    
    if allow_print:
        print(f'\nErkennung von Manöver {maneuver.get_name()}')
    
        for i in range(len(probabilities)):
            print(f'{maneuver_dict[i]:19}: {probabilities[i]}%')
        
    if pred_string == maneuver.get_name():
        print(f'\nVorhersage: \033[92m{pred_string}\033[0m')
        return True
    else:
        print(f'\nVorhersage: \033[31m{pred_string}\033[0m')
        return False
    
    
    
def predict_partial_maneuver(maneuver, draw_plot=True):
    for length in range(300, 1, -1):
        m = maneuver.get_partial(length)
        prediction = pred_partial_single(m, allow_print=False)
        
        if not prediction:
            print(f'Das Manöver wurde mit einer Länge von {min(length + 1, len(maneuver))} noch richtig erkannt ({((min(length + 1, len(maneuver))/len(maneuver)) * 100):.2f}%).')
            
            if draw_plot:
                draw_maneuvers([maneuver.get_partial(min(length + 1, len(maneuver)))])
            return length + 1
    return len(maneuver)



def predict_partial_amount(maneuver, amount, allow_print=True):
    scores = []
    for m in maneuver.generate_maneuvers(amount):
        scores.append(predict_partial_maneuver(m, False))
        
    average_score = sum(scores)/len(scores)
    if allow_print:
        print(f'Das Manöver {maneuver.get_name()} wird bei einer durchschnittlichen Länge von {average_score:.2f} erkannt.')
        print(f'Dies entspricht {average_score/len(maneuver) * 100:.2f}% der Gesamtlänge des Manövers.')
    return average_score
        
        

def predict_partial_all(amount_per_maneuver):
    all_scores = []
    
    for m in maneuvers:
        all_scores.append(predict_partial_amount(m, amount_per_maneuver, allow_print=False))
        
    for i, score in enumerate(all_scores):
        print(f'\nDas Manöver {maneuver_dict[i]} wird bei einer durchschnittlichen Länge von {score:.2f} erkannt.')
        print(f'Dies entspricht {score/300 * 100:.2f}% der Gesamtlänge des Manövers.')
 
 
if __name__ == '__main__':
    predict_partial_all(100)