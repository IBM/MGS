# =================================================================
# Licensed Materials - Property of IBM
#
# "Restricted Materials of IBM
#
# BMC-YKT-08-23-2011-2
#
# (C) Copyright IBM Corp. 2005-2014  All rights reserved
# US Government Users Restricted Rights -
# Use, duplication or disclosure restricted by
# GSA ADP Schedule Contract with IBM Corp.
#
# =================================================================

# The pathname is relative to the lens directory
THIS_DIR := extensions/functor/Zipper
THIS_STEM := Zipper

SRC_PREFIX := $(THIS_DIR)/src
OBJ_PREFIX := $(THIS_DIR)/obj

SOURCES := CG_ZipperInitArgs.C \
CG_ZipperExecArgs.C \
CG_ZipperBase.C \
Zipper.C \
CG_ZipperType.C \
CG_ZipperFactory.C 

# define the full pathname for each file
SRC_$(THIS_STEM) = $(patsubst %,$(SRC_PREFIX)/%, $(SOURCES))

THIS_SRC := $(SRC_$(THIS_STEM))
SRC += $(THIS_SRC)

# Create the list of object files by substituting .C with .o
TEMP :=  $(patsubst %.C,%.o,$(filter %.C,$(THIS_SRC)))

OBJ_$(THIS_STEM) := $(subst src,obj,$(TEMP))
OBJS += $(OBJ_$(THIS_STEM))

EXTENSION_OBJECTS += $(OBJ_$(THIS_STEM))
