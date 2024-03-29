import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_aspect('equal', adjustable='box')

lim_min = 100000000000
lim_max = 0


def plot(maneuver):
    """
    uses a Maneuver to plot it into a 3D graph.

    :return: a set of three arrays holding the coordinates for the matlibplotter
    """
    return ([i.getX() for i in maneuver],
            [i.getY() for i in maneuver],
            [i.getZ() for i in maneuver])


def __draw_help(maneuver):
    global lim_max, lim_min
    # translating maneuver to coordinate points
    xs, ys, zs = plot(maneuver)

    # setting limitations for the plot
    lim_min_tmp = min(min(xs), min(ys), min(zs))
    lim_max_tmp = max(max(xs), max(ys), max(zs))

    if lim_min > lim_min_tmp:
        lim_min = lim_min_tmp
    if lim_max < lim_max_tmp:
        lim_max = lim_max_tmp

    ax.set_xlim3d(lim_min, lim_max)
    ax.set_ylim3d(lim_min, lim_max)
    ax.set_zlim3d(lim_min, lim_max)

    # plot setup
    ax.plot3D(xs, ys, zs)


def draw_maneuvers(maneuvers: list):
    for m in maneuvers:
        __draw_help(m)
    plt.show()


def draw_updated_maneuvers(maneuvers: list):
    global ax
    plt.clf()
    ax = plt.axes(projection='3d')
    for m in maneuvers:
        __draw_help(m)
    plt.show()
