# This is a code generated file, generated using configure.py
# To change any options please rerun ./configure.py with the desired options.
# To build the project execute make -j <number of processes>
.DEFAULT_GOAL:= final

BIN_DIR?=./bin
EXE_FILE=gslparser

PYTHON_INCLUDE_DIR := /Users/kozloski/miniconda3/include/python3.10
PYTHON_LIB := /Users/kozloski/miniconda3/pkgs/python-3.10.16-hb885b13_1/lib
BISON=$(shell which bison)
FLEX=$(shell which flex)

GSLROOT = $(shell pwd)
OPERATING_SYSTEM = $(shell uname)

SCRIPTS_DIR := scripts
DX_DIR := dx
SO_DIR := $(GSLROOT)/so
PARSER_PATH := framework/parser
STD_UTILS_OBJ_PATH := utils/std/obj
TOTALVIEW_LIBPATH := /opt/toolworks/totalview.8.4.1-7/rs6000/lib

CC := clang++
C_COMP := clang++
HAVE_MPI := 1
HAVE_PTHREADS := 1

MAKE64 = 

####################################################################
# Name of all submodules we want to build
# 1. modules from the GSL frameworks
# 2. modules from the extension (i.e. user-defined)
#
# part 1 --> which include framework/...
#               utils/...
FRAMEWORK_MODULES := \
		dca \
		dataitems \
		factories \
		networks \
		parser \
		simulation \
		functors \
            
UTILS_MODULES := \
		std \
		img \
		mnist-master\
		streams \


BASE_MODULES := $(patsubst %,framework/%,$(FRAMEWORK_MODULES))
BASE_MODULES += $(patsubst %,utils/%,$(UTILS_MODULES))

CONSTANT_MODULES :=

EDGE_MODULES :=

INTERFACE_MODULES :=

NODE_MODULES :=


STRUCT_MODULES := CoordsStruct \

TRIGGER_MODULES := UnsignedServiceTrigger \

VARIABLE_MODULES := \

FUNCTOR_MODULES := BidirectConnectNodeSetsFunctor \
        BinomialDist \
        CombineNVPairs \
        ConnectNodeSetsFunctor \
        DstDimensionConstrainedSampler \
        DstRefDistanceModifier \
        DstRefGaussianWeightModifier \
        DstRefSumRsqrdInvWeightModifier \
        DstScaledContractedGaussianWeightModifier \
        DstScaledGaussianWeightModifier \
        ExecuteShell \
        Exp \
        FloatArrayMaker \
        GetDstNodeCoordFunctor \
        GetNodeCoordFunctor \
        GetPostNodeCoordFunctor \
        GetPreNodeCoordFunctor \
        GetPreNodeIndex \
       	IsoSampler \
        IsoSamplerHybrid \
       	GradientLayout \
       	NormalizedGradientLayout \
        LoadMatrix \
        LoadSparseMatrix \
        Log \
        ModifyParameterSet \
        NameReturnValue \
        Neg \
        PolyConnectorFunctor \
        RandomDispersalLayout \
        RefAngleModifier \
        RefDistanceModifier \
        ReversedDstRefGaussianWeightModifier \
        ReversedSrcRefGaussianWeightModifier \
        ReverseFunctor \
        Round \
        Scale \
        ServiceConnectorFunctor \
        SetSourceArrayIndexFunctor \
        SrcDimensionConstrainedSampler \
        SrcRefDistanceModifier \
        SrcRefDoGWeightModifier \
        SrcRefGaussianWeightModifier \
        SrcRefPeakedWeightModifier \
        SrcRefSumRsqrdInvWeightModifier \
        SrcScaledContractedGaussianWeightModifier \
        SrcScaledGaussianWeightModifier \
        Threshold \
        ToroidalRadialSampler \
        UniformDiscreteDist \

# part 2 --> extension/...
# this files list all the modules we want to build
# so that we don't have to modify this Makefile
include Extensions.mk

## hold the relative path to all extension subfolders
EXTENSION_MODULES += $(patsubst %,constant/%,$(CONSTANT_MODULES))
EXTENSION_MODULES += $(patsubst %,edge/%,$(EDGE_MODULES))
EXTENSION_MODULES += $(patsubst %,functor/%,$(FUNCTOR_MODULES))
EXTENSION_MODULES += $(patsubst %,node/%,$(NODE_MODULES))
EXTENSION_MODULES += $(patsubst %,struct/%,$(STRUCT_MODULES))
EXTENSION_MODULES += $(patsubst %,trigger/%,$(TRIGGER_MODULES))
EXTENSION_MODULES += $(patsubst %,variable/%,$(VARIABLE_MODULES))

# those with only header files
SPECIAL_EXTENSION_MODULES += $(patsubst %,interface/%,$(INTERFACE_MODULES))

EXTENSION_MODULES := $(patsubst %,extensions/%,$(EXTENSION_MODULES))
SPECIAL_EXTENSION_MODULES := $(patsubst %,extensions/%,$(SPECIAL_EXTENSION_MODULES))

MODULES := $(BASE_MODULES)
MODULES += $(EXTENSION_MODULES)

SOURCES_DIRS := $(patsubst %,%/src, $(MODULES))
MYSOURCES := $(foreach dir,$(SOURCES_DIRS),$(wildcard $(dir)/*.C))

#IMPORTANT: If you want to ignore some files
#           add them here (just source file names)
MYSOURCES := $(filter-out %MatrixParser.C %ReadImage.C %Ini.C %Img.C %Matrx.C %ImgUtil.C %Wbuf.C %Lzwbuf.C %Pal.C %Bitbuf.C , $(MYSOURCES))
HEADERS_DIRS := $(patsubst %,%/include, $(MODULES))

SOURCES_FILENAME_ONLY :=$(shell for file in $(notdir $(MYSOURCES)); do \
	       echo $${file} ; \
	       done)
PURE_OBJS := $(patsubst %.C, %.o, $(SOURCES_FILENAME_ONLY))

COMMON_DIR := ../common/obj
COMMON_OBJS := $(foreach dir,$(COMMON_DIR), $(wildcard $(dir)/*.o))

OBJS_DIR := obj
OBJS := $(patsubst %, $(OBJS_DIR)/%, $(PURE_OBJS))

vpath %.C $(SOURCES_DIRS)
vpath %.c $(SOURCES_DIRS)
vpath %.h $(HEADERS_DIRS) framework/parser/generated

$(OBJS) : | $(OBJS_DIR)

$(OBJS_DIR):
	mkdir $(OBJS_DIR)

$(BIN_DIR):
	mkdir $(BIN_DIR)
#OBJECTONLYFLAGS is flags that only apply to objects, depend.sh generated code.
OBJECTONLYFLAGS := $(MAKE64)
# OTHER_LIBS :=-lgmp -lpython2.7
OTHER_LIBS := -include utils/std/include/darwin_compat.h -I$(PYTHON_INCLUDE_DIR) -L$(PYTHON_LIB) -lpython3.10 -isysroot
OTHER_LIBS_HEADER :=-I$(PYTHON_INCLUDE_DIR) -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk 

ifeq ($(USE_SUITESPARSE), 1)
OTHER_LIBS += -I$(SUITESPARSE)/include -L$(SUITESPARSE)/lib -lcxsparse
OTHER_LIBS_HEADER += -I$(SUITESPARSE)/include -DUSE_SUITESPARSE
endif
#LDFLAGS := -shared

# Base include paths and features
CFLAGS := -include utils/std/include/darwin_compat.h \
          $(patsubst %,-I %/include,$(MODULES)) \
          $(patsubst %,-I %/generated,$(PARSER_PATH)) \
          $(patsubst %,-I %/include,$(SPECIAL_EXTENSION_MODULES)) \
          -I../common/include \
          -I$(GSLROOT)/utils/std/include \
          -I$(GSLROOT)/utils/streams/include \
          -I/opt/homebrew/opt/gmp/include \
          -I/opt/homebrew/opt/open-mpi/include

# Platform and feature flags
CFLAGS += -DDARWIN -DDISABLE_DYNAMIC_LOADING -DHAVE_MPI -DNOWARNING_DYNAMICCAST

# Compiler options
CFLAGS += -std=c++17 -stdlib=libc++ -O3 -fPIC -MMD 

# Library configuration for C++17
CFLAGS += -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES \
          -D_LIBCPP_BUILDING_LIBRARY \
          -D_LIBCPP_DISABLE_EXTERN_TEMPLATE

# Warning suppressions
CFLAGS += -Wno-deprecated-declarations \
          -Wno-error=c++11-extensions \
          -Wno-error=c++14-extensions \
          -Wno-c++11-narrowing \
          -Wno-deprecated-register \
          -Wno-reserved-user-defined-literal \
          -Wno-incompatible-pointer-types-discards-qualifiers \
          -Wno-implicit-int-conversion

# Use the specific library file path
LIBS := /opt/homebrew/lib/libgmp.a $(LENS_LIBS_EXT) -L/opt/homebrew/opt/open-mpi/lib -lmpi -lmpi_mpifh -Wl,-rpath,/opt/homebrew/opt/open-mpi/lib -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib $(OTHER_LIBS)
FINAL_TARGET_FLAG = -Wl,-export_dynamic,-rpath,/Users/kozloski/miniconda3/pkgs/python-3.10.16-hb885b13_1/lib -v
XLINKER := 

# clear all default suffixes
.SUFFIXES:

# Our Suffix rules
#%.d:
#	@$(SCRIPTS_DIR)/depend.sh $(subst /obj,,${@D}) ${*F} $(CFLAGS) > $@

#.C:
#	true

#.h:
#	true

$(OBJS_DIR)/%.o : %.C
	$(CC) $(CFLAGS) -c $< $(OTHER_LIBS_HEADER) -o $@
# Include the description of each module.
# This will be in the root/project, where root is the root subdir (E.g. dir1)
# include $(patsubst %,%/module.mk,$(MODULES))

# Include the C include dependencies
-include $(OBJS:.o=.d)


# Parser targets
speclang.tab.h:
	cd framework/parser/bison; $(BISON) -v -d speclang.y; \
	mv speclang.tab.c ../generated/speclang.tab.C 2>/dev/null; mv speclang.tab.h ../generated

framework/parser/generated/speclang.tab.C: framework/parser/bison/speclang.y
	cd framework/parser/bison; $(BISON) -v -d speclang.y; \
	mv speclang.tab.c ../generated/speclang.tab.C 2>/dev/null; mv speclang.tab.h ../generated


$(OBJS_DIR)/speclang.tab.o: framework/parser/generated/speclang.tab.C framework/parser/bison/speclang.y
	g++ -c $< -DYYDEBUG $(CFLAGS) $(OBJECTONLYFLAGS) -o $@

# Scanner targets
framework/parser/generated/lex.yy.C: framework/parser/flex/speclang.l
	$(FLEX) -+ framework/parser/flex/speclang.l
	sed 's/class istream;/#include <FlexFixer.h>/'            lex.yy.cc > lex.yy.cc.edited
	mv -f lex.yy.cc.edited framework/parser/generated/lex.yy.C
	rm lex.yy.cc

$(OBJS_DIR)/lex.yy.o: framework/parser/generated/lex.yy.C framework/parser/flex/speclang.l
	g++ -c $< $(CFLAGS) $(OBJECTONLYFLAGS) -o $@

cleanfirst:
	-rm $(BIN_DIR)/$(EXE_FILE)

final: cleanfirst speclang.tab.h $(OBJS) $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o | $(BIN_DIR)
	$(CC) $(FINAL_TARGET_FLAG) -v $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o framework/dca/src/fakesocket.c $(OBJS) $(COMMON_OBJS) $(LDFLAGS) $(LIBS) -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lz -o $(BIN_DIR)/$(EXE_FILE)

$(BIN_DIR)/$(EXE_FILE): cleanfirst $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o $(BASE_OBJECTS) $(LENS_LIBS_EXT)  | $(BIN_DIR)
	$(CC) $(FINAL_TARGET_FLAG) $(NEEDED_OBJS) $(COMMON_OBJS) $(OBJS_DIR)/speclang.tab.o $(OBJS_DIR)/lex.yy.o $(OBJS_DIR)/socket.o $(LIBS) -o $@

.PHONY: clean
clean:
	-rm -f dx/EdgeSetSubscriberSocket
	-rm -f dx/NodeSetSubscriberSocket
	-rm -f $(BIN_DIR)/$(EXE_FILE)
	-rm -f $(BIN_DIR)/createDF
	-rm -f lib/liblens.a
	-rm -f lib/liblensext.a
	-rm -f $(SO_DIR)/Dependfile
	-rm -f framework/parser/bison/speclang.output
	-rm -f framework/parser/generated/lex.yy.C
	-rm -f framework/parser/generated/lex.yy.C
	-rm -f framework/parser/generated/speclang.tab.C
	-rm -f framework/parser/generated/speclang.tab.h
	-rm -f $(OBJS_DIR)/*
