import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Maneuver import Maneuver, State
from FileParser import parse_file


def plot(maneuver):
    """
    uses a Maneuver to plot it into a 3D graph.

    :return: a set of three arrays holding the coordinates for the matlibplotter
    """
    return ([i.getX() for i in maneuver.getNodes()],
            [i.getY() for i in maneuver.getNodes()],
            [i.getZ() for i in maneuver.getNodes()])


maneuver = parse_file("Looping")
#maneuver.getNodes().append(State(0, 0, 0))
xs, ys, zs = plot(maneuver)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ln, = ax.plot(xs, ys, zs)
ax.legend()
ax.set_aspect('equal', adjustable='box')

lim_min = min(min(xs), min(ys), min(zs))
lim_max = max(max(xs), max(ys), max(zs))

# m = maneuver.stretch(0, 0, 100)
# a, b, c = plot(m)
# ax.plot(a, b, c)

for elem in maneuver.generate_maneuvers(5):
     xs, ys, zs = plot(elem)
     lim_min = min(lim_min, min(min(xs), min(ys), min(zs)))
     lim_max = max(lim_max, max(max(xs), max(ys), max(zs)))
     ln, = ax.plot(xs, ys, zs)

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
#     maneuver = maneuver.scale()
#     xs, ys, zs = plot(maneuver)
#
#     ln, = ax.plot(xs, ys, zs)
#
# animation = FuncAnimation(fig, update, interval=500, repeat=True)
plt.show()