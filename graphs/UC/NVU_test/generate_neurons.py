import argparse

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


# def calculateRealCoordinates(index, length, total_nvus):
#     import math
#     return index * length - ((math.sqrt(total_nvus) - 1) * length)

def calculateRealCoordinatesLowerLeftCorner(index, length, nvu_dim):
    """
    index (int): index in the given dimension
    length (float): length of each element in that dimention
    nvu_dim (int): number of element in that dimension
    """
    return index * length - ((nvu_dim - 1) * length)


def calculateRealCoordinatesCenter(index, length, nvu_dim):
    """
    index (int): index in the given dimension
    length (float): length of each element in that dimention
    nvu_dim (int): number of element in that dimension
    """
    return index * length - ((nvu_dim - 1) * length) + length/2


# def calculateRealCoordinates3(indices, length,
#                               NumberNVUs,
#                               # xNumberNVUs, yNumberNVUs, zNumberNVUs
#                               ):
#     result = []
#     for _i in len(indices):
#         c = indices[_i] * 2 * length - (NumberNVUs[_i] - 1) * length
#         result.append(c)
#     return result


def gen_neurons_in_region(region, num_neuron=1, min_distance=0):
    """ generate 'num_neuron' neurons within the 2D region

    region (tuple): (x_l, x_u), (y_l, y_u)

    num_neuron (int): number of neurons in that region

    min_distance (float):  each neuron should not be too-closed to each other, i.e. at least min_distane
    """
    radius = min_distance  # 2.0 # (um)
    rangeX = region[0]
    rangeY = region[1]

    if num_neuron == 1:
        # special case
        randPoints = []
        x_u = rangeX[1]
        x_l = rangeX[0]
        y_u = rangeY[1]
        y_l = rangeY[0]
        randPoints.append((x_l+(x_u-x_l)/2, y_l+(y_u-y_l)/2))
        return randPoints
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
    # print(randPoints)
    return randPoints


def genNeuronFile(args):
    """
    generate the tissue file
    """
    f = open("tissue.txt", 'w')
    f.write("#FILENAME{0}LAYER{0}MTYPE{0}ETYPE{0}XOFFSET{0}YOFFSET{0}ZOFFSET{0}OFFSET_TYPE{0}AXON_PAR{0}BASAL_PAR{0}APICAL_PAR\n".format(delimiter))

    total_nvus = args.x_nvus * args.y_nvus

    for x in range(0, args.x_nvus):  # at NVU x-th along x-axis

        x_offset = calculateRealCoordinatesCenter(x, args.length, args.x_nvus)

        for y in range(0, args.y_nvus):

            y_offset = calculateRealCoordinatesCenter(y, args.length, args.y_nvus)

            for z in range(0, z_nvus):
                # z_offset = calculateRealCoordinatesCenter(z, args.length, args.z_nvus)
                z_offset = calculateRealCoordinatesCenter(z, args.length, z_nvus)

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
                    if (total_nvus * args.neuron_per_nvus < 10):
                        print(x_offset, y_offset, z_offset)
        if (total_nvus * args.neuron_per_nvus < 10):
            print('---------')
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
