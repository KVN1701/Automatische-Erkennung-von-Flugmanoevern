import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from maneuver import Maneuver, State
from helpful_methods import parse_file


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.legend()
ax.set_aspect('equal', adjustable='box')


def plot(maneuver):
    """
    uses a Maneuver to plot it into a 3D graph.

    :return: a set of three arrays holding the coordinates for the matlibplotter
    """
    return ([i.getX() for i in maneuver],
            [i.getY() for i in maneuver],
            [i.getZ() for i in maneuver])


def __draw_help(maneuver):
     # translating maneuver to coordinate points
     xs, ys, zs = plot(maneuver)

     # setting limitations for the plot
     lim_min = min(min(xs), min(ys), min(zs))
     lim_max = max(max(xs), max(ys), max(zs))
     
     ax.set_xlim3d(lim_min, lim_max)
     ax.set_ylim3d(lim_min, lim_max)
     ax.set_zlim3d(lim_min, lim_max)
     
     # plot setup
     ln, = ax.plot(xs, ys, zs)
     
     
def draw_maneuvers(maneuvers: list):
     for m in maneuvers:
          __draw_help(m)
     plt.show()