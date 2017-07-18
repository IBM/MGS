# =================================================================
# Licensed Materials - Property of IBM
#
# "Restricted Materials of IBM
#
# BCM-YKT-07-18-2017
#
# (C) Copyright IBM Corp. 2005-2017  All rights reserved
# US Government Users Restricted Rights -
# Use, duplication or disclosure restricted by
# GSA ADP Schedule Contract with IBM Corp.
#
# =================================================================

# The pathname is relative to the lens directory
THIS_DIR := extensions/functor/TissueMGSifyFunctor
THIS_STEM := TissueMGSifyFunctor

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := CG_TissueMGSifyFunctorInitArgs.C \
CG_TissueMGSifyFunctorExecArgs.C \
CG_TissueMGSifyFunctorBase.C \
TissueMGSifyFunctor.C \
CG_TissueMGSifyFunctorType.C \
CG_TissueMGSifyFunctorFactory.C 

# define the full pathname for each file
SRC_$(THIS_STEM) = $(patsubst %,$(SRC_PREFIX)/%, $(SOURCES))

THIS_SRC := $(SRC_$(THIS_STEM))
SRC += $(THIS_SRC)

# Create the list of object files by substituting .C with .o
TEMP :=  $(patsubst %.C,%.o,$(filter %.C,$(THIS_SRC)))

OBJ_$(THIS_STEM) := $(subst src,obj,$(TEMP))
OBJS += $(OBJ_$(THIS_STEM))

EXTENSION_OBJECTS += $(OBJ_$(THIS_STEM))
