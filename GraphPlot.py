import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Maneuver import Maneuver, State


def plot(maneuver):
    """
    uses a Maneuver to plot it into a 3D graph.

    :return: a set of three arrays holding the coordinates for the matlibplotter
    """
    return ([i.getX() for i in maneuver.getNodes()],
            [i.getY() for i in maneuver.getNodes()],
            [i.getZ() for i in maneuver.getNodes()])


maneuver = Maneuver([State(1, 2, 3), State(4, 5, 6)])
xs, ys, zs = plot(maneuver)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ln, = ax.plot(xs, ys, zs)
ax.legend()
ax.set_aspect('equal', adjustable='box')

lim_min = min(min(xs), min(ys), min(zs))
lim_max = max(max(xs), max(ys), max(zs))
ax.set_xlim3d(lim_min, lim_max)
ax.set_ylim3d(lim_min, lim_max)
ax.set_zlim3d(lim_min, lim_max)


# Test ob plot updaten kann
# def update(frame):
#     global maneuver
#     ax.clear()
#     ax.set_xlim3d(lim_min, lim_max)
#     ax.set_ylim3d(lim_min, lim_max)
#     ax.set_zlim3d(lim_min, lim_max)
#
#     maneuver = maneuver.turn(10)
#     xs, ys, zs = plot(maneuver)
#
#     ln, = ax.plot(xs, ys, zs)
#
# animation = FuncAnimation(fig, update, interval=500, repeat=True)
plt.show()