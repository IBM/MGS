import sys
# Needs 7 input arguments explained below
#This script creates a Tissue File with a list of neurons.swc files and their co-ordinates in X Y Z dimensions
#xoffset, yoffset, zoffset refer to distance between neurons in each directions X, Y, Z respectively
xoffset = float(sys.argv[1])
yoffset = float(sys.argv[2])
zoffset = float(sys.argv[3])

#xdim,ydim,zdim to define size of the grid in 3 dimensions X, Y, Z.
xdim = int(sys.argv[4])
ydim = int(sys.argv[5])
zdim = int(sys.argv[6])

#The swc file which will be used is provided as an argument
filename = sys.argv[7]

#USER DEFINED
#Contents of output file are written to
outfilename = 'neurons_out.txt'
#Initial X, Y, Z positions - Change them here if they don't start at 0,0,0
xinit=0
yinit=0
zinit=0

#Writes the contents to file in a specified format
with open(outfilename,'a') as the_file:
     the_file.write("#FILENAME LAYER MTYPE ETYPE XOFFSET YOFFSET ZOFFSET OFFSET_TYPE AXON_PAR BASAL_PAR APICAL_PAR")
     for k in range(0,zdim):
            for j in range(0,ydim):
                   for i in range(0,xdim):
                             the_file.write('\n%s 1 1 0 %f %f %f R NULL NULL NULL' %(filename,xinit+i*xoffset,yinit+j*yoffset,zinit+k*zoffset))
the_file.close()
