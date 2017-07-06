# The pathname is relative to the lens directory
# Define some common prefixes/suffixes for use
THIS_DIR := framework/dca
THIS_STEM := dca

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES :=  \
BoolTrigger.C \
BoolTriggerDescriptor.C \
BoolTriggerFactory.C \
CompositeTrigger.C \
CompositeTriggerDescriptor.C \
CompositeTriggerFactory.C \
ConnectionSetQueriable.C \
DataItemQueriable.C \
DoubleTrigger.C \
DoubleTriggerDescriptor.C \
DoubleTriggerFactory.C \
EdgeQueriable.C \
EnumEntry.C \
FloatTrigger.C \
FloatTriggerDescriptor.C \
FloatTriggerFactory.C \
GeneratedPublisherBase.C \
GridQueriable.C \
InstanceFactoryQueriable.C \
InstanceFactoryRegistryQueriable.C \
IntTrigger.C \
IntTriggerDescriptor.C \
IntTriggerFactory.C \
NodeQueriable.C \
PublisherQueriable.C \
PublisherRegistry.C \
PublisherRegistryQueriable.C \
Queriable.C \
QueriableDescriptor.C \
QueryDescriptor.C \
QueryField.C \
QueryResult.C \
RepertoireQueriable.C \
SimulationPublisher.C \
SimulationQueriable.C \
TriggerBase.C \
TriggerableBase.C \
TriggeredPauseAction.C \
TriggerWorkUnit.C \
UnsignedTrigger.C \
UnsignedTriggerDescriptor.C \
UnsignedTriggerFactory.C \
SemaphoreTrigger.C \
SemaphoreTriggerDescriptor.C \
SemaphoreTriggerFactory.C \

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
