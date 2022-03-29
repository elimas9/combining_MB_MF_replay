from scipy.spatial import Voronoi
from PIL import Image
import matplotlib as mpl
from optparse import OptionParser
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_latex_bridge as mlb
import json

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)
mlb.figure_textwidth(0.45)

INF = 100000000000000000


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
    #--------------------------------------------------------------------
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    #--------------------------------------------------------------------
    new_regions = []
    new_vertices = vor.vertices.tolist()
    #--------------------------------------------------------------------
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    #--------------------------------------------------------------------
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in list(zip(vor.ridge_points, vor.ridge_vertices)):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    #--------------------------------------------------------------------
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        #----------------------------------------------------------------
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        #----------------------------------------------------------------
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        #----------------------------------------------------------------
        for p2, v1, v2 in ridges:
            #------------------------------------------------------------
            if v2 < 0:
                v1, v2 = v2, v1
            #------------------------------------------------------------
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            #------------------------------------------------------------
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            #------------------------------------------------------------
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            #------------------------------------------------------------
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        #----------------------------------------------------------------
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        #----------------------------------------------------------------
        # finish
        new_regions.append(new_region.tolist())
    #--------------------------------------------------------------------
    return new_regions, np.asarray(new_vertices)
    #--------------------------------------------------------------------


def create_map(img, offset, resolution, file1, file2, keystates,iteration, num, entropies, action):
    """
    Create the map with color code according to the expert which choose the action in this state
    """
    x_file = file1[:,1]
    y_file = file1[:,2]

    # make up data points
    x = x_file
    y = y_file
    xmin = (x.min() - offset[0]) / resolution
    xmax = (x.max() - offset[0]) / resolution
    ymin = (y.min() - offset[1]) / resolution
    ymax = (y.max() - offset[1]) / resolution
    xc = (file2[:,1] - offset[0]) / resolution
    yc = (file2[:,2] - offset[1]) / resolution
    pointsForVoronoi = list(zip(xc, yc))

    # compute Voronoi tesselation
    vor = Voronoi(pointsForVoronoi)

    # reconstruct infnit voronoi
    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig, ax = plt.subplots()

    # Change the goal after a certain iteration
    if iteration < keystates[2]:
        goal = keystates[1]
    else:
        goal = keystates[3]

    # colorize polygon according to the expert who choose the action to do
    state = 0

    # normalize the entropy values
    normalizer = mpl.colors.Normalize(vmin=np.min(entropies), vmax=np.max(entropies))

    for region in regions:
        polygon = vertices[region]
        colorState = plt.cm.hot(normalizer(entropies[state]))
        plt.fill(*list(zip(*polygon)), color=colorState, alpha=1, zorder=0)
        state = state + 1

    # Create the map
    marker_size = 4

    for s in np.arange(len(xc)):
        if s in keystates[0]:
            color = 'green'
            label = 'initial states'
        elif s == goal:
            color = 'purple'
            label = 'first reward state'
        elif s == keystates[3]:
            color = 'orange'
            label = 'second reward state'
        else:
            color = 'grey'
            label = 'states'
        ax.plot(xc[s], yc[s], c=color, marker = 'o', markersize=marker_size, label=label, zorder=3, linestyle="None")
        text_color = "black"

        ax.text(xc[s]+4, yc[s]-1, str(s), color=text_color , zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    margin = 30
    ax.axis([xmin-margin, xmax+margin, ymin-margin, ymax+margin])

    plt.tick_params(axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False)

    plt.tick_params(axis='y',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False)

    ax.imshow(img, zorder=2, cmap=plt.cm.hot)
    plt.axis("off")

    plt.savefig('output_maps/map'+str(num)+'.png')
    plt.show()
    plt.close()


def manage_keystates_file(keyStatesFile):
    """
    Manage the file that contains the key states
    """
    init = list()
    with open(keyStatesFile,'r') as file3:
        for line in file3:
            if line.split(" ")[0] == "goal":
                goal = int(line.split(" ")[1])
            elif line.split(" ")[0] == "new_goal":
                switch = int(line.split(" ")[1])
                new_goal = int(line.split(" ")[2])
            elif line.split(" ")[0] == "init":
                init = init + list(map(int, line.split(" ")[1:]))
    keystates = [init, goal, switch, new_goal]
    return keystates


def manage_map_files(mapDataPath):
    """
    Manage the files requisite for construct the map
    """
    onlyfiles = [f for f in listdir(mapDataPath) if isfile(join(mapDataPath,f))]
    poseOverTime = [s for s in onlyfiles if "poseCell_log" in s][0]
    statePositions = [s for s in onlyfiles if "voronoiCenters_exp" in s][0]
    expGlobalParam = [s for s in onlyfiles if "param_exp" in s][0]
    mapsOverTime = sorted([s for s in onlyfiles if ".yaml" in s])
    runData = mapDataPath + "/" + poseOverTime
    file1 = np.genfromtxt(runData)
    stateData = mapDataPath + "/" + statePositions
    file2 = np.genfromtxt(stateData)
    paramDict = {}
    with open(mapDataPath + "/" + mapsOverTime[-1]) as f:
        for line in f:
            try:
                (key, val) = line.split(":")
            except ValueError:
                break
            paramDict[key] = val[1:].strip('\n')
    paramGlobalDict = {}
    with open(mapDataPath + "/" + expGlobalParam) as f:
        for line in f:
            try:
                (key, val) = line.split(":")
            except ValueError:
                break
            paramGlobalDict[key] = val[1:].strip('\n')
    # apply transparency on image
    img = Image.open(mapDataPath +"/"+ paramDict['image'])
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if (0 <= item[0] <= 150) and (0 <= item[1] <= 150) and (0 <= item[2] <= 150):
             # draw the border in grey
            newData.append((100, 100, 100, 255))
        else:
            # just apply the transparency, don't care about the color
            newData.append((0, 0, 0, 0)) 

    img.putdata(newData)
    resolution = float(paramDict['resolution'])
    offset = ((paramDict['origin'][1:(len(paramDict['origin'])-1)]).replace(" ","")).split(',')
    offset = [float(i) for i in offset]
    return img, offset, resolution, file1, file2


if __name__ == "__main__":
    usage = "usage: plotColoredMapByMaxEntropyBeforeSwitch.py [options] [path toward map files]" \
            "[file that contains the key states] [file that contains the dynamics of the experts]" \
            "[name of output file without the extension]"
    parser = OptionParser(usage)


    mapDataPath = sys.argv[1]
    keyStatesFile = sys.argv[2]
    outputFile = sys.argv[3]

    params = {
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False,
    'figure.figsize': [10.0, 10.0]
    }


    ##############################  Fetch entropies ####################################################################
    DIVIDE = {}


    with open('realisticWorld', 'r') as file1: #/!\ THE RIGHT FILE HAS TO BE SELECTED, TO GET THE PROPER ENTROPIES!
        arena = json.load(file1)
        for state in arena["transitionActions"]:
            state_actions_dividers = np.zeros(8)
            for transition in state['transitions']:
                state_actions_dividers[int(transition['action'])] += int(transition['prob'])
            DIVIDE[state['state']] = state_actions_dividers[:]

    nb_states = len(DIVIDE)

    PROBAS = np.zeros((8, nb_states, nb_states))
    with open('realisticWorld', 'r') as file1:
        arena = json.load(file1)
        for state in arena["transitionActions"]:
            for transition in state['transitions']:
                PROBAS[int(transition['action'])][int(state['state'])][int(transition['state'])] = int(
                    transition['prob']) / DIVIDE[state['state']][int(transition['action'])]

    ENTROPIES = {}
    mine = 10
    maxe = 0
    for a in range(len(PROBAS)):
        entropies = []
        for s in range(len(PROBAS[a])):
            entropie = 0
            for p in PROBAS[a][s]:
                if p != 0:
                    entropie -= p * np.log2(p)
            entropies.append(entropie)
            if entropie > maxe:
                maxe = entropie
            if entropie < mine:
                mine = entropie
        ENTROPIES[str(a)] = entropies[:]

    print(" Min and Max entropies among all actions and all states: ", mine, maxe)


    # --------------- Code to get the max entropy for all states -------------------
    #----------- Max entropy for each state ----------------
    MAX = []
    MAXa = []
    for S in range(len(ENTROPIES['0'])):
        max = 0
        maxa = '0'
        for a in ENTROPIES:
            for s in range(len(ENTROPIES[a])):
                if s == int(S):
                    if ENTROPIES[a][s] > max:
                        max = ENTROPIES[a][s]
                        maxa = int(a)

        MAX.append(max)
        MAXa.append(maxa)

    MAX = MAX


    img, offset, resolution, file1, file2 = manage_map_files(mapDataPath)
    keystates = manage_keystates_file(keyStatesFile)


    it = 0  # The time step of the environment
    num = "_max_before_switch"  # the id of the map in the output folder
    create_map(img, offset, resolution, file1, file2, keystates, it, num, MAX, action=-1)  # For map for max entropy



