# The pathname is relative to the lens directory
# Define some common prefixes/suffixes for use
THIS_DIR := framework/simulation
THIS_STEM := simulation

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := ISender.C \
IReceiver.C \
FinalPhase.C \
InitPhase.C \
LENSServer.C \
LoadPhase.C \
Pauser.C \
Phase.C \
RuntimePhase.C \
Simulation.C \
Stopper.C \
ThreadPool.C \
PhaseElement.C \

#ifeq ($(HAVE_PTHREADS), 1)
#
#SOURCES += LENSServer.C \
#ThreadPool.C 
#
#endif

# Checkpoint.C\
# Storable.C \

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


