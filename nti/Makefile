
.DEFAULT_GOAL: target

SRC_DIR :=./src
OBJ_DIR :=./obj
HEADER_DIR:=./include

vpath %.h $(HEADER_DIR)
vpath %.cxx $(SRC_DIR)

FOR_BLUEGENE_P=0
FOR_BLUEGENE_L=0
FOR_BLUEGENE_Q=0
FOR_AIX=0
A2AW=0

BIN_DIR=./bin


ifeq ($(A2AW), 1)

A2A = -DA2AW

endif



ifeq ($(FOR_BLUEGENE_Q), 1)

CXX=mpixlcxx_r
LD=mpixlcxx_r
BGQ_FLOOR = /bgsys/drivers/ppcfloor
LDFLAGS = 
CFLAGS = $(A2A) -DUSING_BLUEGENE -DUSING_BLUEGENEQ #-DDISABLE_PTHREADS 
CFLAGS += -O2 -qarch=qp -qtune=qp -qdebug=unlimitin

else

ifeq ($(FOR_BLUEGENE_P), 1)

CXX=mpixlcxx_r
LD=mpixlcxx_r
BGP_FLOOR = /bgsys/drivers/ppcfloor
LDFLAGS = 
CFLAGS = $(A2A) -DUSING_BLUEGENE -DUSING_BLUEGENEP #-DDISABLE_PTHREADS 
CFLAGS += -I$(BGP_FLOOR)/arch/include -O2 -qarch=450 -qtune=450 -qdebug=unlimitin

else

ifeq ($(FOR_BLUEGENE_L), 1)

#CVC=/gpfs/S_fs1/kozloski/bc/viz/bg/cvc/
MPI_INCDIRS=-I/bgl/BlueLight/ppcfloor/bglsys/include
MPI_LIBDIRS=-L/bgl/BlueLight/ppcfloor/bglsys/lib
MPI_LIBS=-lmpich.rts -lmsglayer.rts -ldevices.rts -lrts.rts -ldevices.rts -lrts.rts
CXX=blrts_xlC
LD=blrts_xlC
CFLAGS = $(A2A) -DUSING_BLUEGENE -DUSING_BLUEGENEL -DMPICH_IGNORE_CXX_SEEK $(MPI_INCDIRS) #-I$(CVC)/include/CVC
LDFLAGS = $(MPI_LIBDIRS) #-L$(CVC)/lib/CVC
#EXTRA_LD = -lcvcmc -lcvct
#CVCSRC = cvc.cxx

else

ifeq ($(FOR_AIX), 1)

CXX=mpCC_r
LD=mpCC_r
MAKE64=#-q64
CFLAGS = $(A2A) -DUAIX #-DDISABLE_PTHREADS

else

#CVC=/home/kozloski/bc/viz/linux/cvc/
CXX=mpiCC
LD=mpiCC
#LDFLAGS = -L$(CVC)/lib/CVC
#EXTRA_LD = -lcvcmc -lcvct
CFLAGS = $(A2A)  #-DDISABLE_PTHREADS -I$(CVC)/include/CVC
#CFLAGS = -DUSING_BLUEGENE -I$(CVC)/include/CVC -DVERBOSE
#CVCSRC = cvc.cxx
CFLAGS  += -MMD

endif

endif

endif

endif

STD_UTILS_OBJ_PATH := ../gsl/utils/std/obj
STD_UTILS_INC_PATH := ../gsl/utils/std/include/
CFLAGS += -I$(HEADER_DIR) 
#Add support for GMP library
CFLAGS += -lgmp -std=c++11 
LDFLAGS += -lgmp -std=c++11 -I../common/include -I./include -I$(STD_UTILS_INC_PATH) \
					 -I../gsl/framework/factories/include/


DEFINES = #-DVERBOSE -DBINARY64BITS
ifeq ($(debug), yes)
OPT=
DEBUG=-g
else
OPT=-O3
DEBUG=
endif

UTILS=-I../gsl/utils/std/include -I../common/include/
COPYRIGHT=-I../gsl/framework/factories/include

NG_SHARED_SOURCES = CompositeSwc.cxx \

NG_UNSHARED_SOURCES = neuroGen.cxx \

NG_SOURCES = Neurogenesis.cxx \
NeurogenParams.cxx \
NeurogenSegment.cxx \
NeurogenBranch.cxx \
BoundingCuboid.cxx \
BoundingSurfaceMesh.cxx \

ND_SHARED_SOURCES = Branch.cxx \
SegmentKeySegmentSpace.cxx \
NeuroDevCommandLine.cxx \
Capsule.cxx \
Communicator.cxx \
Director.cxx \
History.cxx \
Neuron.cxx \
NeuronPartitioner.cxx \
NeuroDevParser.cxx \
Params.cxx \
Segment.cxx \
SegmentDescriptor.cxx \
VolumeDecomposition.cxx \
Tissue.cxx \
TissueSlicer.cxx \

ND_UNSHARED_SOURCES = neuroDev.cxx \
AllInTouchSpace.cxx
#ND_SOURCES = $(CVCSRC)

ND_SOURCES =  FrontLimitedSegmentSpace.cxx \
FrontSegmentSpace.cxx \
NeuroDevTissueSlicer.cxx \
SegmentForce.cxx \
SegmentForceAggregator.cxx \
SegmentForceDetector.cxx \
Touch.cxx \

TD_UNSHARED_SOURCES = touchDetect.cxx \

TD_SOURCES = AllInTouchSpace.cxx \
AntiredundancyTouchFilter.cxx \
LENSTissueSlicer.cxx \
DendriteTouchSpace.cxx \
InferiorOliveGlomeruliDetector.cxx \
NonautapticTouchSpace.cxx \
NonautapticAxodendriticTouchSpace.cxx \
ORTouchSpace.cxx \
PassThroughTouchFilter.cxx \
Rotation.cxx \
SynapseTouchSpace.cxx \
Touch.cxx \
TouchAggregator.cxx \
TouchAnalyzer.cxx \
TouchDetector.cxx \
TouchDetectTissueSlicer.cxx \
TouchTable.cxx \
TouchVector.cxx \
Translation.cxx \
ZeroTouchAnalysis.cxx \
TissueContext.cxx \

TISSUE_LIB_SOURCES = CompartmentKey.cxx \

ND_SHARED_OBJECTS = $(patsubst %, $(OBJ_DIR)/%, ${ND_SHARED_SOURCES:.cxx=.o})

NG_SHARED_OBJECTS = $(patsubst %, $(OBJ_DIR)/%, ${NG_SHARED_SOURCES:.cxx=.o})

NG_OBJECTS = $(NG_SHARED_OBJECTS) $(patsubst %, $(OBJ_DIR)/%,${NG_UNSHARED_SOURCES:.cxx=.o}) $(patsubst %, $(OBJ_DIR)/%, ${NG_SOURCES:.cxx=.o})

ND_OBJECTS = $(ND_SHARED_OBJECTS) $(NG_SHARED_OBJECTS) $(patsubst %, $(OBJ_DIR)/%, ${ND_UNSHARED_SOURCES:.cxx=.o}) $(patsubst %, $(OBJ_DIR)/%, ${ND_SOURCES:.cxx=.o})

TD_OBJECTS = $(ND_SHARED_OBJECTS) $(NG_SHARED_OBJECTS) $(patsubst %, $(OBJ_DIR)/%, ${TD_UNSHARED_SOURCES:.cxx=.o}) $(patsubst %, $(OBJ_DIR)/%, ${TD_SOURCES:.cxx=.o})

TISSUE_LIB_OBJECTS = $(ND_SHARED_OBJECTS) $(NG_SHARED_OBJECTS) $(patsubst %, $(OBJ_DIR)/%, ${NG_SOURCES:.cxx=.o}) $(patsubst %, $(OBJ_DIR)/%, ${ND_SOURCES:.cxx=.o}) $(patsubst %, $(OBJ_DIR)/%, ${TD_SOURCES:.cxx=.o}) $(patsubst %, $(OBJ_DIR)/%, ${TISSUE_LIB_SOURCES:.cxx=.o})

COMMON_DIR=../common/obj/
COMMON_OBJS := $(foreach dir,$(COMMON_DIR), $(wildcard $(dir)/*.o))

TISSUE_LIB = ../gsl/lib/liblensdomain.a
#TISSUE_LIB = ../lib/libTissueSim.a
TARGETS = $(TISSUE_LIB) neuroGen neuroDev touchDetect

target: $(TARGETS)

depend: ; makedepend $(ND_SOURCES)

uninstall: 

install: all

clean: ; -rm -f $(ND_SHARED_OBJECTS) $(NG_SHARED_OBJECTS) $(NG_OBJECTS) $(ND_OBJECTS) $(TD_OBJECTS) TissueContext.o
realclean: clean
	-rm -f $(TARGETS) 

#.SUFFIXES: .cxx .cpp .c .h .o
# Include the C include dependencies
-include $(OBJS:.o=.d)


$(OBJ_DIR): 
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o : %.cxx | $(OBJ_DIR)
	$(CXX) -I. -c $< $(DEBUG) $(OPT) $(CFLAGS) $(UTILS) $(COPYRIGHT) $(MAKE64) -o $@

%.o : %.C  | $(OBJ_DIR)
	$(CXX) -I. -c $< $(DEBUG) $(OPT) $(CFLAGS) $(UTILS) $(COPYRIGHT) $(MAKE64) -o ../gsl/utils/std/src $*.C

%.cxx : | $(SRC_DIR)

$(NG_OBJECTS): | $(OBJ_DIR)

$(ND_OBJECTS): | $(OBJ_DIR)

$(TD_OBJECTS): | $(OBJ_DIR)

neuroGen: $(NG_OBJECTS)
	mkdir -p $(BIN_DIR)
	$(LD) $(MAKE64) $(DEBUG) -o $(BIN_DIR)/neuroGen $(OPT) \
		$(NG_OBJECTS) \
		$(LDFLAGS) \
		$(EXTRA_LD) $(MPI_LIBS)

neuroDev: $(ND_OBJECTS)
	mkdir -p $(BIN_DIR)
	$(LD) $(MAKE64) $(DEBUG) -o $(BIN_DIR)/neuroDev $(OPT) \
		$(ND_OBJECTS) \
		$(COMMON_OBJS)\
		$(LDFLAGS) \
		$(EXTRA_LD) $(MPI_LIBS)

touchDetect: $(TD_OBJECTS)
	mkdir -p $(BIN_DIR)
	$(LD) $(MAKE64) $(DEBUG) -o $(BIN_DIR)/touchDetect $(OPT) \
		$(TD_OBJECTS) \
		$(COMMON_OBJS) \
		$(LDFLAGS) \
		$(EXTRA_LD) $(MPI_LIBS)

$(TISSUE_LIB): $(TISSUE_LIB_OBJECTS)
		ar rcs $(TISSUE_LIB) \
		$(TISSUE_LIB_OBJECTS)

docs:

co:
	co $(RCSFILES) 

# DO NOT DELETE THIS LINE -- make depend depends on it.

