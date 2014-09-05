# The pathname is relative to the lens directory
# Define some common prefixes/suffixes for use
THIS_DIR := framework/networks
THIS_STEM := networks

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := Constant.C \
ConstantBase.C \
Composite.C \
CompCategoryBase.C \
ConnectionSet.C \
EdgeBase.C \
EdgeCompCategoryBase.C \
EdgePartitionItem.C \
EdgeSet.C \
FFTGranuleMapper.C \
FFTGranuleMapperFactory.C \
FFTGranuleMapperType.C \
Granule.C \
GranuleConnection.C \
GranuleMapperBase.C \
Graph.C \
GraphConnection.C \
Grid.C \
GridGranuleMapper.C \
GridGranuleMapperFactory.C \
GridGranuleMapperType.C \
GridLayerData.C \
GridLayerDescriptor.C \
GridSet.C \
NodeAccessor.C \
NodeBase.C \
NodeCompCategoryBase.C \
NodeInstanceAccessor.C \
NodePartitionItem.C \
NodeProxyBase.C \
NodeSet.C \
NodeTypeSet.C \
OneToOnePartitioner.C \
BGCartesianPartitioner.C \
RankGranuleMapper.C \
RankGranuleMapperFactory.C \
RankGranuleMapperType.C \
ReadGraphPartitioner.C \
Repertoire.C \
SeparationConstraint.C \
StridesList.C \
Struct.C \
StructDemarshallerBase.C \
IndexSet.C \
Variable.C \
VariableBase.C \
VariableCompCategoryBase.C \
VariableGranuleMapper.C \
VariablePartitionItem.C \
VariableInstanceAccessor.C \
VolumeDivider.C \
VolumeGranuleMapper.C \
VolumeGranuleMapperFactory.C \
VolumeGranuleMapperType.C \

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
