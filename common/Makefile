# =============================================================================
# (C) Copyright IBM Corp. 2005-2025. All rights reserved.
#
# Distributed under the terms of the Apache License
# Version 2.0, January 2004.
# (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
#
# =============================================================================
OBJDIR= ./obj
SRCDIR= ./src
INCDIR= ./include
CC=g++
CFLAGS=-I$(INCDIR) -Wall -Werror -std=c++17 -Wno-deprecated-declarations # -pedantic
CFLAGS+= -fPIC 
ifeq ($(USE_SUITESPARSE), 1)
   OTHER_LIBS := -I$(SUITESPARSE)/include -L$(SUITESPARSE)/lib -lcxsparse -DUSE_SUITESPARSE
endif
LDFLAGS+= -shared

RELEASE?="" # Potentially set by make_XXX or the user from the command line

TARGET_LIB_DIR=../gsl/lib
# Check if release mode
ifeq ($(RELEASE), -release)
   # No debugging
   # Yes optimization
   CFLAGS += -O3
   UTILS_LIB_SHARED = $(TARGET_LIB_DIR)/libutils.so
   UTILS_LIB = $(TARGET_LIB_DIR)/libutils.a
else
   # Yes debugging
   CFLAGS += -ggdb -g
   # Minimal optimization
   CFLAGS += -Og
   UTILS_LIB_SHARED = $(TARGET_LIB_DIR)/libutils_db.so
   UTILS_LIB = $(TARGET_LIB_DIR)/libutils_db.a
endif

ifeq ($(shell uname), Darwin)
    # Add Darwin-specific include paths
   CFLAGS += -I/opt/homebrew/include -I/opt/homebrew/opt/gmp/include
   LDFLAGS += -L/opt/homebrew/lib -lgmp -stdlib=libc++ -lc++
   # Add necessary runtime path
   LDFLAGS += -Wl,-rpath,/opt/homebrew/lib -L/opt/homebrew/opt/open-mpi/lib -lmpi
else
   CFLAGS += -lgmp 
   LDFLAGS += -lgmp 
endif

.DEFAULT_GOAL := final

SRCS=$(foreach dir, $(SRCDIR), $(wildcard $(dir)/*.C))
TMPS=$(foreach dir, $(SRCDIR), $(patsubst $(dir)/%, $(OBJDIR)/%, $(wildcard $(dir)/*.C)))
OBJS=$(patsubst %.C, %.o, $(TMPS))
#OBJS=$(patsubst $($(PURE_SRCS:.C=.o )

COLAB_FILES:=  FileUtils.C NumberUtils.C StringUtils.C
LEFT_SRCS := $(filter-out $(foreach file, ${COLAB_FILES}, %$(file)), $(SRCS))
COLAB_FILES_OBJS=$(patsubst %.C, ./obj/%.o, $(COLAB_FILES))


vpath %.C $(SRCDIR)
vpath %.h $(INCDIR)

$(OBJS): | $(OBJDIR)

$(OBJDIR): 
	mkdir -p $(OBJDIR)

final: $(OBJS) library
	@echo $(OBJS)

$(OBJDIR)/%.o: %.C 
	$(CC) $(OTHER_LIBS) $(CFLAGS) -c $< -o $@

clean: 
	-@rm -f $(OBJDIR)/*

realclean: clean
	-@rm -f $(UTILS_LIB) $(UTILS_LIB_SHARED)

library: ${COLAB_FILES_OBJS}
	@mkdir -p $(TARGET_LIB_DIR)
	ar rcs $(UTILS_LIB) \
	   $(COLAB_FILES_OBJS)
	$(CC) $(LDFLAGS) -o ${UTILS_LIB_SHARED} \
	   $(COLAB_FILES_OBJS)
