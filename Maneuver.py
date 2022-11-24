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
    def __init__(self, nodes: list):
        """
        Creates an instance of a Maneuver. The position of the plane is defined
        by a list of Vectors containing the plane's position (maybe also other information).

        :param nodes: list of States defining the entire graph to symbolise the maneuver.
        """
        self.__nodes = nodes


    def getNodes(self):
        """
        :return: Returns the list of States that describe the Maneuver
        """
        return self.__nodes


    def getTotalTime(self):
        """
        :return: Returns the total time the Maneuver needed
        """
        return self.__nodes[-1].getTime()


    def randomize(self, max_inv: float=10):
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


    def turn(self, angle: float=random.randrange(0, 360 * 2) / 2): # Zufällige Gradzahl in 0.5er Schritten
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


    def stretch(self, factor: float=random.randrange(-20, 20) / 2): # zufälliger Faktor zwischen -10% und 10% in 0,5er Schritten
        """
        Sretches or shrinks a Maneuver by a given factor.

        :param factor: The resizing factor in percent
        :return: the resized Maneuver
        """
        tmp = [self.__nodes[0]] # der erste Punkt wird übernommen
        for index in range(1, len(self.__nodes)):
            node = self.__nodes[index]
            prev_node = self.__nodes[index - 1]
            x, y, z = node.getX() - prev_node.getX(), node.getY() - prev_node.getY(), node.getZ() - prev_node.getZ()
            x, y, z = x + (factor / 100) * x, y + (factor / 100) * y, z + (factor / 100) * z # den Abstand der Vektoren anhand factor anpassen

            tmp_x, tmp_y, tmp_z = tmp[index - 1].getCoordinates()
            tmp.append(State(tmp_x + x, tmp_y + y, tmp_z + z, node.getRotation(), node.getTime()))
        return Maneuver(tmp)


    def generate_maneuvers(self, amount: int, max_inv=None, factor=None, angle=None):
        """
        Used to create random Maneuvers based of the current Maneuver by using the implementeded methods.

        :param amount: The amount of random Maneuvers, that will be created
        :param angle: The angle in which the Maneuvers should be turned
        :param factor: The factor by whicht the Maneuvers should be enlarged or shortened
        :param max_inv: the max amount of movement of the points
        :return: a list of Maneuvers
        """
        tmp = []
        for _ in range(amount):
            m = self.stretch(factor) if factor is not None else self.stretch()
            m = m.turn(angle) if angle is not None else m.turn()
            m = m.randomize(max_inv) if max_inv is not None else m.randomize()
            tmp.append(m)
        return tmp



    def getTotalDistance(self):
        tmp = 0
        for index in range(len(self.__nodes)):
            tmp += math.fabs(self.__nodes[index] - self.__nodes[index + 1])
        return tmp
