import math
import random


class State:
    def __init__(self, x, y, z, rot=None, time=None):
        """
        Creates an instance of State. States represent the individual points of
        a maneuver.

        :param x: x-coordinate
        :param y: y-coordinate
        :param z: z-coordinate
        :param rot: the rotation value of the plane
        :param time: the time when the plane was in this certain state
        """
        self.__x = x
        self.__y = y
        self.__z = z
        self.__rot = rot
        self.__time = time

    def getX(self):
        """
        :return: returns the x coordinate saved in the State
        """
        return self.__x

    def getY(self):
        """
        :return: returns the y coordinate saved in the State
        """
        return self.__y

    def getZ(self):
        """
        :return: returns the z coordinate saved in the State
        """
        return self.__z

    def getCoordinates(self):
        """
        :return: returns the coordinates saved in the State
        """
        return self.__x, self.__y, self.__z

    def getRotation(self):
        """
        :return: returns the rotation saved in the State
        """
        return self.__rot

    def getTime(self):
        """
        :return: returns the time saved in the State
        """
        return self.__time

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__x == other.__x and \
                   self.__y == other.__y and \
                   self.__z == other.__z and \
                   self.__time == other.__time
        return False

    def randomize(self, max_inv): #TODO: Check if step is needed
        """
        Replaces the State to a random position by a maximum value of max_inv

        :param max_inv: maximal distance in meters
        :return: the position of the new State
        """
        x = self.__x + random.randrange(-max_inv, max_inv)
        y = self.__y + random.randrange(-max_inv, max_inv)
        z = self.__z + random.randrange(-max_inv, max_inv)
        return State(x, y, z, self.__rot, self.__time)


class Maneuver:
    def __init__(self, nodes):
        """
        Creates an instance of a Maneuver. The position of the plane is defined
        by a list of Vectors containing the plane's position (maybe also other information).

        :param nodes: list of States defining the entire graph to symbolise the maneuver.
        """
        self.__nodes = nodes


    def getNodes(self):
       return self.__nodes


    def getTotalTime(self):
        return self.__nodes[-1].getTime()


    def randomize(self, max_inv):
        """
        Returns a random generated Maneuver. Every point is moved in a random distance from
        it's original location depending on max_inv.
        :param max_inv: maximal distance in meters
        :return: Maneuver
        """
        tmp = []
        for n in self.__nodes:
            tmp.append(n.randomize(max_inv))
            #TODO: set rotation in state
        return Maneuver(tmp)


    def turn(self, angle=random.randrange(0, 360 * 2) / 2): # Zufällige Gradzahl in 0.5er Schritten
        """
        Turns the Maneuver by a random degree (0° to 360°)

        :param angle: angle which can be added to get a predefined angle
        :return: A new, turned Maneuver
        """
        # z-Achse ist die vertikale, y und x müssen für eine Rotation angepasst werden
        rad_angle = (math.pi / 180) * angle # umrechnen in Radiant

        headx, heady, headz = self.__nodes[0].getCoordinates()
        tailx, taily, tailz = self.__nodes[-1].getCoordinates()
        c_x, c_y, _ = ((headx + tailx) / 2, (heady + taily) / 2, (headz + tailz) / 2) # Punkt um den rotiert werden soll (Mittelpunkt)

        tmp = []
        for n in self.__nodes:
            n_x, n_y, n_z = n.getCoordinates()
            # Anwenden der Rotationsmatrix
            tmp_x = (math.cos(rad_angle) * (n_x - c_x) + math.sin(rad_angle) * (n_y - c_y)) + c_x
            tmp_y = (-math.sin(rad_angle) * (n_x - c_x) + math.cos(rad_angle) * (n_y - c_y)) + c_y
            tmp.append(State(tmp_x, tmp_y, n_z, n.getRotation(), n.getTime())) #TODO: Rotation muss noch geupdated werden
        return Maneuver(tmp)


    def strech(self, factor=random.randrange(-20, 20) / 2): # zufälliger Faktor zwischen -10% und 10% in 0,5er Schritten
        """
        Sretches or shrinks a Maneuver by a given factor.

        :param factor: The resizing factor in percent
        :return: the resized Maneuver
        """



    def getTotalDistance(self):
        tmp = 0
        for index in range(len(self.__nodes)-1):
            tmp += math.fabs(self.__nodes[index] - self.__nodes[index + 1])
        return tmp
