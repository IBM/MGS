import argparse
import math

delimiter = "\t"

# These will likely change to be dyamic/params/something interesting
layer = 1
mtype = 0
etype = 0
offset_type = "A"
axon_par = "NULL"
basal_par = "NULL"
apical_par = "NULL"
z_nvus = 1


def calculateRealCoordinates(index, length, total_nvus):
    return index * 2 * length - ((math.sqrt(total_nvus) - 1) * length)


def gen_neurons_in_region(region, num_neuron=1, min_distance=0):
    """ generate 'num_neuron' neurons within the region (x,y,z) as tuple
    each neuron should not be too-closed to each other, i.e. at least min_distane
    """
    radius = min_distance  # 2.0 # (um)
    rangeX = region[0]
    rangeY = region[1]
    # Generate a set of all points within 200 of the origin, to be used as offsets later
    # There's probably a more efficient way to do this.
    deltas = set()
    for x in range(-radius, radius+1):
        for y in range(-radius, radius+1):
            if x*x + y*y <= radius*radius:
                deltas.add((x, y))

    randPoints = []
    excluded = set()
    i = 0
    qty = num_neuron
    import random
    while i < qty:
        x = random.randrange(*rangeX)
        y = random.randrange(*rangeY)
        if (x, y) in excluded:
            continue
        randPoints.append((x, y))
        i += 1
        excluded.update((x+dx, y+dy) for (dx, dy) in deltas)
    print(randPoints)
    return randPoints


def genNeuronFile(args):
    """
    generate the tissue file
    """
    f = open("tissue.txt", 'w')
    f.write("#FILENAME{0}LAYER{0}MTYPE{0}ETYPE{0}XOFFSET{0}YOFFSET{0}ZOFFSET{0}OFFSET_TYPE{0}AXON_PAR{0}BASAL_PAR{0}APICAL_PAR\n".format(delimiter))

    total_nvus = args.x_nvus * args.y_nvus

    for x in range(0, args.x_nvus):  # at NVU x-th along x-axis

        x_offset = calculateRealCoordinates(x, args.length, total_nvus)

        for y in range(0, args.y_nvus):

            y_offset = calculateRealCoordinates(y, args.length, total_nvus)

            for z in range(0, z_nvus):
                z_offset = 0.0

                scale_factor = 0.9  # <= 1 so that it is not too closed to the border
                half = scale_factor * args.length
                ix = int(half/2)
                # region = ((-half/2, +half/2), (-half/2, +half/2))
                region = ((-ix, +ix), (-ix, +ix))
                min_distance = 3  # um
                randPoints = gen_neurons_in_region(region, args.neuron_per_nvus, min_distance=min_distance)
                for _i in range(0, args.neuron_per_nvus):
                    x_offset += randPoints[_i][0]
                    y_offset += randPoints[_i][1]
                    f.write(
                        args.swc_path + delimiter +
                        str(layer) + delimiter +
                        str(mtype) + delimiter +
                        str(etype) + delimiter +
                        str(x_offset) + delimiter +
                        str(y_offset) + delimiter +
                        str(z_offset) + delimiter +
                        str(offset_type) + delimiter +
                        str(axon_par) + delimiter +
                        str(basal_par) + delimiter +
                        str(apical_par) + "\n"
                        )
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('swc_path', type=str, help="Relative path to the .swc file definition for a neuron, e.g. ./neurons/scptmsn.swc")
    parser.add_argument('length', type=float, help="The length of a single NVU in micro-meters, e.g. 200")
    parser.add_argument('x_nvus', type=int, help="The number of NVU's along the X-axis.")
    parser.add_argument('y_nvus', type=int, help="The number of NVU's along the Y-axis.")
    parser.add_argument('neuron_per_nvus', type=int, default=1, help="The number of neurons per NVU (default = 1).")

    args = parser.parse_args()
    genNeuronFile(args)
