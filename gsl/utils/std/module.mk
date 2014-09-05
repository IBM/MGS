# The pathname is relative to the lens directory
# Define some common prefixes/suffixes for use
THIS_DIR := utils/std
THIS_STEM:= std

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := CheckerboardOdometer.C \
BitMapHeader.C \
CachedPrimeSieve.C \
Connector.C \
CommandLine.C \
C_rand.C \
GranuleConnector.C \
MatrixParser.C \
NoConnectConnector.C \
NDPair.C \
NDPairList.C \
NDPairItemFinder.C \
ParameterSet.C \
QueryPathParser.C \
RandomOrderOdometer.C \
SurfaceOdometer.C \
SysTimer.C \
VolumeOdometer.C \
LensConnector.C \
MaxFloatFullPrecision.C \
VectoredOdometer.C \
VectoredCheckerboardOdometer.C \
VectorOstream.C \
Sigmoid.C \
ShuffleDeck.C \
String.C \
Parser.C \
PhaseDelay.C \
Pearson.C \
TypeClassifier.C \
#MersenneTwister.C \

# define the full pathname for each file
SRC_$(THIS_STEM) = $(patsubst %,$(SRC_PREFIX)/%, $(SOURCES))
THIS_SRC := $(SRC_$(THIS_STEM))
SRC += $(THIS_SRC)

# Create the list of object files by substituting .C with .o
TEMP :=  $(patsubst %.C,%.o,$(filter %.C,$(THIS_SRC)))

# Since the .o files will be in the directory 'obj', while the
# source is in the directory 'source', make this substitution.
# See Gnu Make documentation, section entitled 'Functions for 
# string substition and analysis'
# E.g this creates 'OBJ_datacollect', on which the libdatacollect will depend
OBJ_$(THIS_STEM) := $(subst src,obj,$(TEMP))

OBJS += $(OBJ_$(THIS_STEM))
BASE_OBJECTS += $(OBJ_$(THIS_STEM))
