from Maneuver import Maneuver, State
from math import floor
import numpy as np

"""
Holds some methods that could be useful in multiple files
"""


def parse_file(file_name: str):
    """
    Creates a new Maneuver from a given File holding its coordinates

    :param file_name: name of the file
    :return: an instance of Maneuver
    """
    file = open(f'assets/{file_name}', 'r')
    string_data = file.readlines()
    file.close()

    coordinates = []
    for line in string_data:
        line = line.replace("(", "").replace(")", "")
        tmp = [float(val) for val in line.split(", ")]
        coordinates.append(State(tmp[0], tmp[2],tmp[1])) # has to be switched, because unity has a different coord system
    return Maneuver(coordinates)



def format_time(time_in_seconds: float) -> str:
    """
    Reformates a given second value to hours, minutes and seconds.
    
    :param time_in_seconds: the time value in seconds
    :return: a String containing the formated time value
    """
    hours = floor(time_in_seconds/3600)
    minutes = floor((time_in_seconds - hours * 3600)/60)
    seconds = time_in_seconds - minutes * 60 - hours * 3600
    if minutes > 0:
        if hours > 0:
            return f'{hours} hours {minutes} minutes and {seconds:.2f} seconds'
        return f'{minutes} minutes and {seconds:.2f} seconds'
    return f'{seconds:.2f} seconds'



def generate_dataset(amount, maneuver_list):
    """
    Generates a dataset to train the Neural Network.
    
    :param amount: the amount of training data
    :param maneuver_list: a list of maneuvers to generate the data from
    :return: the dataset in a tuple containing the data and labels
    """
    num_of_maneuvers = len(maneuver_list)
    
    # generating the maneuvers to train/ test the neural network
    x_dataset = []
    tmp = []
    for i in range(len(maneuver_list)): 
        m = maneuver_list[i]
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