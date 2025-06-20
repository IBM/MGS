# The pathname is relative to the gsl directory
# Define some common prefixes/suffixes for use
THIS_DIR := framework/functors
THIS_STEM := functorss

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

#FunctorType.C \
#FunctorFactory.C \

SOURCES := \
BindBackFunctor.C \
BindBackFunctorType.C \
BindBackFunctorFactory.C \
BindFrontFunctor.C \
BindFrontFunctorType.C \
BindFrontFunctorFactory.C \
BindNameFunctor.C \
BindNameFunctorType.C \
BindNameFunctorFactory.C \
ConnectionScriptFunctor.C \
ConnectorFunctor.C \
ConnectSets2Functor.C \
ConnectSets2FunctorType.C \
ConnectSets2FunctorFactory.C \
EdgeInitializerFunctor.C \
EachAvgFunctor.C \
EachAvgFunctorType.C \
EachAvgFunctorFactory.C \
EachDstFunctor.C \
EachDstFunctorType.C \
EachDstFunctorFactory.C \
EachDstPropSrcFunctor.C \
EachDstPropSrcFunctorType.C \
EachDstPropSrcFunctorFactory.C \
EachSrcFunctor.C \
EachSrcFunctorType.C \
EachSrcFunctorFactory.C \
EachSrcPropDstFunctor.C \
EachSrcPropDstFunctorType.C \
EachSrcPropDstFunctorFactory.C \
Functor.C \
GaussianFunctor.C \
GaussianFunctorType.C \
GaussianFunctorFactory.C \
UniformDistFunctor.C \
UniformDistFunctorType.C \
UniformDistFunctorFactory.C \
InAttrDefaultFunctor.C \
InAttrDefaultFunctorType.C \
InAttrDefaultFunctorFactory.C \
InAttrInitializerFunctor.C \
OutAttrDefaultFunctor.C \
OutAttrDefaultFunctorType.C \
OutAttrDefaultFunctorFactory.C \
OpenCircleLayoutFunctor.C \
OpenCircleLayoutFunctorType.C \
OpenCircleLayoutFunctorFactory.C \
LayoutFunctor.C \
ManhattanRingFunctor.C \
ManhattanRingFunctorType.C \
ManhattanRingFunctorFactory.C \
ManhattanRing2Functor.C \
ManhattanRing2FunctorType.C \
ManhattanRing2FunctorFactory.C \
NodeDefaultFunctor.C \
NodeDefaultFunctorType.C \
NodeDefaultFunctorFactory.C \
EdgeDefaultFunctor.C \
EdgeDefaultFunctorType.C \
EdgeDefaultFunctorFactory.C \
NDPairListFunctor.C \
NdplEdgeInitFunctor.C \
NdplEdgeInitFunctorType.C \
NdplEdgeInitFunctorFactory.C \
NdplInAttrInitFunctor.C \
NdplInAttrInitFunctorType.C \
NdplInAttrInitFunctorFactory.C \
NdplNodeInitFunctor.C \
NdplNodeInitFunctorType.C \
NdplNodeInitFunctorFactory.C \
NdplModifierFunctor.C \
NdplModifierFunctorType.C \
NdplModifierFunctorFactory.C \
NodeInitializerFunctor.C \
PrintFunctor.C \
PrintFunctorType.C \
PrintFunctorFactory.C \
RadialDensitySamplerFunctor.C \
RadialDensitySamplerFunctorType.C \
RadialDensitySamplerFunctorFactory.C \
RadialHistoSamplerFunctor.C \
RadialHistoSamplerFunctorType.C \
RadialHistoSamplerFunctorFactory.C \
RadialSamplerFunctor.C \
RadialSamplerFunctorType.C \
RadialSamplerFunctorFactory.C \
RangePassThruFunctor.C \
RangePassThruFunctorType.C \
RangePassThruFunctorFactory.C \
RefPtGenFunctor.C \
SameFunctor.C \
SameFunctorType.C \
SameFunctorFactory.C \
SampFctr1Functor.C \
SampFctr2Functor.C \
SubNodeSetFunctor.C \
SumFunctor.C \
SumFunctorType.C \
SumFunctorFactory.C \
TraverseFunctor.C \
TraverseFunctorType.C \
TraverseFunctorFactory.C \
UniformLayoutFunctor.C \
UniformLayoutFunctorType.C \
UniformLayoutFunctorFactory.C \
UniqueFunctor.C \
UniqueFunctorType.C \
UniqueFunctorFactory.C \

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
