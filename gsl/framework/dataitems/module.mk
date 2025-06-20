# The pathname is relative to the gsl directory
# Define some common prefixes/suffixes for use
THIS_DIR := framework/dataitems
THIS_STEM := dataitems

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := ArrayDataItem.C\
BoolArrayDataItem.C\
BoolDataItem.C\
CharArrayDataItem.C\
CharDataItem.C\
ComplexDataItem.C \
ConstantDataItem.C \
ConstantTypeDataItem.C \
ConnectionSetDataItem.C\
DataItem.C\
DataItemArrayDataItem.C\
DataItemException.C\
DoubleArrayDataItem.C\
DoubleDataItem.C\
EdgeDataItem.C\
EdgeSetDataItem.C\
FloatArrayDataItem.C\
FloatDataItem.C\
FunctorDataItem.C\
FunctorTypeDataItem.C\
GranuleMapperDataItem.C\
GranuleMapperTypeDataItem.C\
GridDataItem.C\
GridSetDataItem.C\
IndexSetDataItem.C\
InstanceFactoryDataItem.C\
InstanceFactoryRegistryDataItem.C\
IntArrayDataItem.C\
IntDataItem.C\
LongArrayDataItem.C\
LongDataItem.C\
NDPairDataItem.C\
NDPairListDataItem.C\
NodeSetArrayDataItem.C\
NodeSetDataItem.C\
NodeDataItem.C \
NodePairDataItem.C \
NodeTypeDataItem.C\
NodeTypeSetDataItem.C\
EdgeTypeDataItem.C\
NumericDataItem.C\
NumericArrayDataItem.C\
PublisherDataItem.C\
PublisherRegistryDataItem.C\
ParameterSetDataItem.C\
PhaseDataItem.C\
ReferredInstanceFactoryDataItem.C\
RepertoireDataItem.C\
RepertoireFactoryDataItem.C \
RelativeNodeSetDataItem.C \
ScriptFunctorTypeDataItem.C\
ShortArrayDataItem.C\
ShortDataItem.C\
ServiceDataItem.C\
SignedCharArrayDataItem.C\
SignedCharDataItem.C\
SimulationDataItem.C\
StringArrayDataItem.C\
StridesListDataItem.C\
StringDataItem.C\
StructDataItem.C\
StructPointerDataItem.C\
StructTypeDataItem.C\
TriggerDataItem.C\
TriggerTypeDataItem.C\
UnsignedCharArrayDataItem.C\
UnsignedCharDataItem.C\
UnsignedIntArrayDataItem.C\
UnsignedIntDataItem.C\
UnsignedShortArrayDataItem.C\
UnsignedShortDataItem.C \
VariableDataItem.C \
VariableTypeDataItem.C \
#StructArrayDataItem.C\

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
