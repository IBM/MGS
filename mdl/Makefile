# Run this file through gnu make, /usr/gnu/bin/make 

default: bin/mdlparser

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
# e.g. Use gmake ADDCFLAGS='-g' if you want to compile with the -g option.
ADDCFLAGS :=
CFLAGS += $(ADDCFLAGS)

OPERATING_SYSTEM = $(shell uname)

# Can add extra system libraries later if needed
LIBS :=

# Each module adds to this initial empty definition of SRC
SRCS :=

FINAL_TARGET_FLAG := 

ifeq ($(CC), g++)
  CFLAGS += -g -Wall -Wno-unused -Wpointer-arith -Wcast-qual -Wcast-align
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

#For debugging
ifeq ($(CC), g++)
   CFLAGS += -ggdb
endif
ifeq ($(CC), xlC_r)
   CFLAGS += -g
endif

#Optimization
#CFLAGS += -O2

# include sources
include sources.mk

# Parser targets
mdl.tab.h: parser/mdl.y
	cd parser; bison -v -d  mdl.y; mv mdl.tab.c mdl.tab.C; cd ..

parser/mdl.tab.C: parser/mdl.y
	cd parser; bison -v -d  mdl.y; mv mdl.tab.c mdl.tab.C; cd ..

# Scanner targets
ifeq ($(FLEXFIXER), True)
scanner/lex.yy.C: scanner/mdl.l
	flex -+ scanner/mdl.l
	sed "s/class istream;/#include <FlexFixer.h>/" \
	lex.yy.cc > lex.yy.cc.edited 
	mv -f lex.yy.cc.edited scanner/lex.yy.C
	- rm lex.yy.cc
else
scanner/lex.yy.C: scanner/mdl.l
	flex -+ scanner/mdl.l
	mv -f lex.yy.cc scanner/lex.yy.C
endif

scanner/lex.yy.o: scanner/lex.yy.C scanner/mdl.l
	$(CC) -c $< $(CFLAGS) -o $@

# Include the C include dependencies
include $(OBJS:.o=.d)

# Our Suffix rules
%.d: 
	scripts/depend.sh ${*F} $(CFLAGS) > $@

.C:
	true	

.h:
	true

lib/libmdl.a: $(OBJS)
	ar rvu $@ $(OBJS)
	ranlib $@	

LIBS += lib/libmdl.a


bin/mdlparser: parser/mdl.tab.C lib/libmdl.a scanner/lex.yy.C
	$(CC) $(CFLAGS) parser/mdl.tab.C scanner/lex.yy.C $(LIBS) \
           $(XLINKER) -DYYDEBUG -o $@ $(FINAL_TARGET_FLAG)

.PHONY: clean
clean:
	- find . -name "*\.o" -exec /bin/rm {} \;
	- find . -name "*\.a" -exec /bin/rm {} \;
	- find . -name "*\.d" -exec /bin/rm {} \;
	- rm bin/mdlparser
	- rm parser/mdl.output
	- rm parser/mdl.tab.C
	- rm parser/mdl.tab.h
	- rm scanner/lex.yy.C

