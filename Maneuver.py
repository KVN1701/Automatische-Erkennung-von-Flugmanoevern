import math
from random import randrange, choice, uniform
import numpy as np


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
        return self.__x

    def getY(self):
        return self.__y

    def getZ(self):
        return self.__z

    def getCoordinates(self):
        return self.__x, self.__y, self.__z

    def getRotation(self):
        return self.__rot

    def getTime(self):
        return self.__time

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__x == other.__x and \
                   self.__y == other.__y and \
                   self.__z == other.__z and \
                   self.__time == other.__time
        return False

    def randomize(self, max_inv):  # TODO: Check if step is needed
        """
        Replaces the State to a random position by a maximum value of max_inv.

        :param max_inv: maximal distance in meters
        :return: the position of the new State
        """
        x = self.__x + uniform(-max_inv, max_inv)
        y = self.__y + uniform(-max_inv, max_inv)
        z = self.__z + uniform(-max_inv, max_inv)
        return State(x, y, z, self.__rot, self.__time)
    
    
    def get_numpy_array(self):
        # return np.array([self.__x, self.__y, self.__z, self.__rot, self.__time])
        return np.array([self.__x, self.__y, self.__z])



class Maneuver:
    def __init__(self, nodes: list):
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


    def __getCenter(self):
        """
        :return: A state which is in the middle of the Maneuver
        """
        min_x, min_y, min_z, max_x, max_y, max_z = 0, 0, 0, 0, 0, 0
        for n in self.__nodes:
            if n.getX() < min_x: min_x = n.getX()
            if n.getX() > max_x: max_x = n.getX()
            if n.getY() < min_y: min_y = n.getY()
            if n.getY() > max_y: max_y = n.getY()
            if n.getZ() < min_z: min_z = n.getZ()
            if n.getZ() > max_z: max_z = n.getZ()
        return (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2


    def randomize(self, max_inv: float = round(uniform(0, 1.75), 2)):
        """
        Returns a random generated Maneuver. Every point is moved in a random distance from
        it's original location depending on max_inv.

        :param max_inv: maximal distance in meters
        :return: Maneuver
        """
        tmp = [n.randomize(max_inv) for n in self.__nodes]
        return Maneuver(tmp)


    def turn(self, angle: float = randrange(0, 360 * 4) / 4):  # Zufällige Gradzahl in 0.5er Schritten
        """
        Turns the Maneuver by a random degree (0° to 360°)

        :param angle: angle which can be added to get a predefined angle
        :return: A new, turned Maneuver
        """
        # z-Achse ist die vertikale, y und x müssen für eine Rotation angepasst werden
        rad_angle = (math.pi / 180) * angle  # umrechnen in Radiant

        headx, heady, headz = self.__nodes[0].getCoordinates()
        tailx, taily, tailz = self.__nodes[-1].getCoordinates()
        c_x, c_y, _ = ((headx + tailx) / 2, (heady + taily) / 2, (headz + tailz) / 2)  # Punkt um den rotiert werden soll (Mittelpunkt)

        tmp = []
        for n in self.__nodes:
            n_x, n_y, n_z = n.getCoordinates()
            # Anwenden der Rotationsmatrix
            tmp_x = (math.cos(rad_angle) * (n_x - c_x) + math.sin(rad_angle) * (n_y - c_y)) + c_x
            tmp_y = (-math.sin(rad_angle) * (n_x - c_x) + math.cos(rad_angle) * (n_y - c_y)) + c_y
            tmp.append(State(tmp_x, tmp_y, n_z, n.getRotation(), n.getTime()))  # TODO: Rotation muss noch geupdated werden
        return Maneuver(tmp)


    def scale(self, factor: float = randrange(-40, 40) / 4):  # zufälliger Faktor zwischen -10% und 10% in 0,5er Schritten
        """
        Sretches or shrinks a Maneuver by a given factor.

        :param factor: The resizing factor in percent
        :return: the resized Maneuver
        """
        tmp = [self.__nodes[0]]  # der erste Punkt wird übernommen
        for index in range(1, len(self.__nodes)):
            node = self.__nodes[index]
            prev_node = self.__nodes[index - 1]
            x, y, z = node.getX() - prev_node.getX(), node.getY() - prev_node.getY(), node.getZ() - prev_node.getZ()
            x, y, z = x + (factor / 100) * x, y + (factor / 100) * y, z + (factor / 100) * z  # den Abstand der Vektoren anhand factor anpassen

            tmp_x, tmp_y, tmp_z = tmp[index - 1].getCoordinates()
            tmp.append(State(tmp_x + x, tmp_y + y, tmp_z + z, node.getRotation(), node.getTime()))
        return Maneuver(tmp)


    def stretch(self,
                factor_x: float = randrange(-40, 40) / 2,
                factor_y: float = randrange(-40, 40) / 2,
                factor_z: float = randrange(-40, 40) / 2):
        """
        Streches the Maneuver in x, y and z direction.

        :param factor_x: the factor by which the Maneuver will be streched in x-direction in percent
        :param factor_y: the factor by which the Maneuver will be streched in y-direction in percent
        :param factor_z: the factor by which the Maneuver will be streched in z-direction in percent
        :return: an instance of Maneuver
        """
        center_x, center_y, center_z = self.__getCenter()
        tmp = []
        for n in self.__nodes:
            x = n.getX() - center_x # distance to the center (x-coordinate)
            y = n.getY() - center_y # distance to the center (y-coordinate)
            z = n.getZ() - center_z # distance to the center (z-coordinate)
            x, y, z = x + (factor_x / 100) * x, y + (factor_y / 100) * y, z + (factor_z / 100) * z
            tmp.append(State(x, y, z, n.getTime(), n.getRotation()))
        return Maneuver(tmp)


    def move(self, 
             distance_x: float = randrange(-800, 800) / 2,
             distance_y: float = randrange(-800, 800) / 2,
             distance_z: float = randrange(-800, 800) / 2):
        """
        Moves the graph in x, y and z direction.
        
        :param distance_x: movement x
        :param distance_y: movement y
        :param distance_z: movement z
        :return: an instance of Maneuver
        """
        return Maneuver([State(
            n.getX() + distance_x,
            n.getY() + distance_y,
            n.getZ() + distance_z,
            n.getTime(),
            n.getRotation()
        ) for n in self.__nodes])


    def mirror(self, mirror: bool = True):
        """
        Mirrors the Maneuver

        :param mirror True if the graph should be mirrored
        :return: an instance of Maneuver
        """
        if mirror:
            _, c_y, _ = self.__getCenter() # centerx, centery, centerz
            tmp = []
            for n in self.__nodes:
                # z-Achse oben -> bleibt gleich
                y = (n.getY() - c_y) * -2
                tmp.append(State(n.getX(), n.getY() + y, n.getZ(), n.getTime(), n.getRotation()))
            return Maneuver(tmp)
        return self


    def generate_maneuvers(self, 
                           amount: int, 
                           max_inv=None, 
                           factor=None, 
                           angle=None, 
                           mirror=True, 
                           dist_move_x=None, 
                           dist_move_y=None, 
                           dist_move_z=None,
                           dist_stretch_x=None,
                           dist_stretch_y=None,
                           dist_stretch_z=None,
                ) -> list:
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
            rand_angle = randrange(0, 360 * 4) / 4
            rand_inv = round(uniform(0, 1.75), 2)
            rand_scale = randrange(-40, 40) / 4
            mirror = choice([True, False]) if mirror else False
            move_x, move_y, move_z = randrange(-800, 800) / 2, randrange(-800, 800) / 2, randrange(-800, 800) / 2
            stretch_x, stretch_y, stretch_z = randrange(-40, 40) / 2, randrange(-40, 40) / 2, randrange(-40, 40) / 2
            
            move_defined = move_x != None and move_y != None and move_z != None
            stretch_defined = stretch_x != None and stretch_y != None and stretch_z != None
            
            # TODO: neu erstellte Methoden einfügen
            m = self.scale(factor) if factor is not None else self.scale(rand_scale)
            m = m.turn(angle) if angle is not None else m.turn(rand_angle)
            m = m.randomize(max_inv) if max_inv is not None else m.randomize(rand_inv)
            m = m.mirror(mirror)
            m = m.move(move_x, move_y, move_z) if move_defined else m.move(dist_move_x, dist_move_y, dist_move_z)
            m = m.stretch(stretch_x, stretch_y, stretch_z) if stretch_defined else m.stretch(dist_stretch_x, dist_stretch_y, dist_stretch_z)
            tmp.append(m)
        return tmp


    def getTotalDistance(self):
        tmp = 0
        for index in range(len(self.__nodes)):
            tmp += math.fabs(self.__nodes[index] - self.__nodes[index + 1])
        return tmp


    def get_numpy_array(self):
        """
        :return: a numpy array holding the data for keras to process.
        """
        return np.array([state.get_numpy_array() for state in self.__nodes])
    
    
    def __len__(self):
        return len(self.__nodes)