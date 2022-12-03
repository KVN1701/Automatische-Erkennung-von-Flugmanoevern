from Maneuver import Maneuver


def parse_file(file_name: str):
    """
    Creates a new Maneuver from a given File holding its coordinates

    :param file_name: name of the file
    :return: an instance of Maneuver
    """
    file = open(f"assets/{file_name}", "r")
    string_data = file.readlines()
    file.close()

    coordinates = []
    for line in string_data:
        line = line.replace("(", "")
        line = line.replace(")", "")
        tmp = [int(val) for val in line.split(", ")]
        coordinates.append((tmp[0], tmp[1],tmp[2]))
    return Maneuver(coordinates)

