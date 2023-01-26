from maneuver import Maneuver, State
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
    return Maneuver(coordinates, name=file_name)



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
            return f'{hours} h {minutes} min {seconds:.2f} sec'
        return f'{minutes} min {seconds:.2f} sec'
    return f'{seconds:.2f} sec'


# The list of Maneuvers the project currently holds
maneuvers = [
    parse_file("Looping"),
    parse_file("LangsamerJoJo"),
    parse_file("SchnellerJoJo"),
    parse_file("Abschwung"),
    parse_file("Kertwende"),
]

# dictionary which links an index to a Maneuver
maneuver_dict = {
    0 : 'Looping',
    1 : 'LangsamerJoJo',
    2 : 'SchnellerJoJo',
    3 : 'Abschwung',
    4 : 'Kertwende',
}

# True if mirroring the maneuver should be allowed

def generate_dataset(amount):
    """
    Generates a dataset to train the Neural Network.
    
    :param amount: the amount of training data
    :return: the dataset in a tuple containing the data and labels
    """
    
    # generating the maneuvers to train/ test the neural network
    x_dataset = []
    tmp = []
    for i in range(len(maneuvers)):
        m = maneuvers[i]
        tmp.extend(m.generate_maneuvers(amount))
        
    # converting the maneuvers to numpy arrays
    for m in tmp:
        x_dataset.append(m.get_numpy_array())   
    x_dataset = np.array(x_dataset, dtype=float)
    
    # generating the labels for the dataset
    y_dataset = []
    for i in range(len(maneuvers)):
        y_dataset.extend([i for _ in range(amount)])
    y_dataset = np.array(y_dataset, dtype=int)
    
    return x_dataset, y_dataset