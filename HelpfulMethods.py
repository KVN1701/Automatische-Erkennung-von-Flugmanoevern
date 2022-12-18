from Maneuver import Maneuver, State
from math import floor

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
