from tensorflow import keras
from helpful_methods import parse_file, generate_dataset, maneuver_dict, maneuvers
import numpy as np
from graph_plot import *
from maneuver import *
from texttable import Texttable


# The model that will be used
model = keras.models.load_model('best_model.h5')


def test_KI(amount):
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
    
    
    
def predict_single(maneuver, draw_plot=True, allow_print=True):
    """
    Gives a prediction of a single maneuver displaying the percentage the
    maneuver resembles a maneuver
    
    :param maneuver: the maneuver that will be tested
    :param draw_plot: True if the maneuver should be displayed in a plot
    :param allow_print: True if the output should be printed
    :return: True if the maneuver was predicted correctly
    """
    # drawing the plot
    if draw_plot:
        draw_maneuvers([maneuver])
        
    # ? When a partial maneuver should be displayed there has to be a normal test before it
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
            print(f'\nVorhersage: \033[92m{pred_string}\033[0m')
        return True
    else:
        if allow_print:
            print(f'\nVorhersage: \033[31m{pred_string}\033[0m')
        return False
    
    
    
def standard_test(amount):
    """
    The standard test consisting of a single prediction of every maneuver.
    
    :param amount: the testing amount
    """
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
            predict_single(m, draw_plot=True)
    

    
def predict_partial_maneuver(maneuver, draw_plot=True):
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
            print(f'Das Manöver wurde mit einer Länge von {min(length + 1, len(maneuver))} noch richtig erkannt ({((min(length + 1, len(maneuver))/len(maneuver)) * 100):.2f}%).')
            
            if draw_plot:
                print('\nIn blau dargestellt ist der Teil des Manövers, der notwendig war, um das Manöver zu erkennen.')
                print('Das komplette Manöver wird daneben in orange als Referenz abgebildet.\n')
                draw_m = maneuver.get_partial(length + 1)
                draw_updated_maneuvers([draw_m, maneuver.move(0, 400, 0)])
                
            return length + 1
    return len(maneuver)



def predict_partial_amount(maneuver, amount, allow_print=True, draw_plot=True):
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
        
    average_score = sum(scores)/len(scores)
    if allow_print:
        print(f'Das Manöver {maneuver.get_name()} wird bei einer durchschnittlichen Länge von {average_score:.2f} erkannt.')
        print(f'Dies entspricht {average_score/len(maneuver) * 100:.2f}% der Gesamtlänge des Manövers.')
    return average_score
        
        

def predict_partial_all(amount_per_maneuver, draw_plot=True):
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
        rows.append([maneuver_dict[i], f'{min(score, 300):.2f}', f'{min(score, 300)/300 * 100:.2}'])
    
    t = Texttable()
    t.add_rows(rows)
    print(t.draw())
 

if __name__ == '__main__':
    predict_partial_all(1000)
    # standard_test(50)