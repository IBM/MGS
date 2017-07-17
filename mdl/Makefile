# Run this file through gnu make, /usr/gnu/bin/make 

default: final 

OBJS_DIR=obj
SOURCES_DIRS=src
BIN_DIR=bin
EXE_FILE=mdlparser

RELEASE?="" # Potentially set by make_XXX or the user from the command line

# To compile with xlC_r edit this or use CC=xlC_r at command line.
CC := g++

# To build code with a fix for Flex (observed on AIX) define the following to True.
FLEXFIXER := True

# clear all default suffixes
.SUFFIXES:

#OBJECTONLYFLAGS is flags that only apply to objects, depend.sh generated code.
OBJECTONLYFLAGS :=

# Look for include files in each of the modules
#  This will append CFLAGS with a string like '-Idir1/include -Idir2/include'
CFLAGS := -Iinclude -Iparser 

# Create a variable 'ADDCFLAGS' that you can use to add to the CFLAGS from the
# command line option of gmake if necessary
# e.g. Use gmake ADDCFLAGS='-g -ggdb' if you want to compile with the -g option.
ADDCFLAGS :=
CFLAGS += $(ADDCFLAGS)


OPERATING_SYSTEM = $(shell uname)

# Can add extra system libraries later if needed
LIBS :=

# Each module adds to this initial empty definition of SRC
SRCS :=

FINAL_TARGET_FLAG := 

ifeq ($(CC), g++)
  CFLAGS += -Wall -Wno-unused -Wpointer-arith -Wcast-qual -Wcast-align
endif

ifeq ($(OPERATING_SYSTEM), Linux)
   CFLAGS +=  -DLINUX
   ifeq ($(CC), g++)
      CFLAGS += -fPIC
   endif
endif
ifeq ($(OPERATING_SYSTEM), AIX)
   ifeq ($(CC), g++)
      XLINKER := -Xlinker -bmaxdata:0x80000000
      CFLAGS += -mminimal-toc -DAIX
   endif
   ifeq ($(CC), xlC_r)
      XLINKER := -bmaxdata:0x80000000 
      CFLAGS += -DAIX
      OBJECTONLYFLAGS := -qrtti=all
      FINAL_TARGET_FLAG := -qstaticinline $(OBJECTONLYFLAGS)
   endif
endif

# Check if release mode
ifeq ($(RELEASE), -release)
   # No debugging
   # Yes optimization
   CFLAGS += -O3
else
   # Yes debugging
   ifeq ($(CC), g++)
      CFLAGS += -ggdb -g
   endif
   ifeq ($(CC), xlC_r)
      CFLAGS += -g
   endif
   # Minimal optimization
   CFLAGS += -Og
endif

vpath  %.C $(SOURCES_DIRS)

# include sources
include sources.mk


#depend: $(SRCS)
#	rm -f ./.depend
#	$(CC) $(CFLAGS) -MM $^>>./.depend;
#
#-include .depend

# Parser targets
mdl.tab.h:  parser/mdl.y
	cd parser; bison -v -d -b mdl mdl.y; cd ..

#parser/mdl.tab.c: parser/mdl.y
#	cd parser; bison -v -d -b mdl mdl.y; cd ..

# Scanner targets
ifeq ($(FLEXFIXER), True)
scanner/lex.yy.cc: scanner/mdl.l
	flex  scanner/mdl.l
	sed "s/class istream;/#include <FlexFixer.h>/" \
	lex.yy.cc > lex.yy.cc.edited 
	mv -f lex.yy.cc.edited scanner/lex.yy.cc
	- rm lex.yy.cc
else
scanner/lex.yy.cc: scanner/mdl.l
	flex  scanner/mdl.l
	mv -f lex.yy.cc scanner/lex.yy.cc
endif

scanner/lex.yy.o: scanner/lex.yy.cc scanner/mdl.l
	$(CC) -c $< $(CFLAGS) -o $@

# Include the C include dependencies
#include $(OBJS:.o=.d)

# Our Suffix rules
#%.d: 
#	scripts/depend.sh ${*F} $(CFLAGS) > $@

$(OBJS_DIR)/%.o : %.C
	$(CC) $(CFLAGS)  $(OBJECTONLYFLAGS) -c $< -o $@


lib/libmdl.a: $(OBJS)
	ar rvu $@ $(OBJS)
	ranlib $@	

LIBS += lib/libmdl.a


final: mdl.tab.h $(OBJS) parser/mdl.tab.c lib/libmdl.a scanner/lex.yy.cc
	-rm $(BIN_DIR)/$(EXE_FILE)
	$(CC) $(CFLAGS) parser/mdl.tab.c scanner/lex.yy.cc $(LIBS) \
           $(XLINKER) -o $(BIN_DIR)/$(EXE_FILE) $(FINAL_TARGET_FLAG)

.PHONY: clean
clean:
	-rm $(OBJS_DIR)/*
	-rm $(BIN_DIR)/$(EXE_FILE)
	-rm parser/mdl.output
	-rm parser/mdl.tab.c
	-rm parser/mdl.tab.h
	-rm scanner/lex.yy.cc


