OBJDIR= ./obj
SRCDIR= ./src
INCDIR= ./include
CC=g++
CFLAGS=-I$(INCDIR) -Wall -Werror -lgmp -std=c++11 -Wno-deprecated-declarations # -pedantic
CFLAGS+= -fPIC 
LDFLAGS+= -shared

RELEASE?="" # Potentially set by make_XXX or the user from the command line

# Check if release mode
ifeq ($(RELEASE), -release)
   # No debugging
   # Yes optimization
   CFLAGS += -O3
   UTILS_LIB_SHARED = ../gsl/lib/libutils.so
   UTILS_LIB = ../gsl/lib/libutils.a
else
   # Yes debugging
   CFLAGS += -ggdb -g
   # Minimal optimization
   CFLAGS += -Og
   UTILS_LIB_SHARED = ../gsl/lib/libutils_db.so
   UTILS_LIB = ../gsl/lib/libutils_db.a
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
	
final: $(OBJS)
	echo $(OBJS)

$(OBJDIR)/%.o: %.C 
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	-@rm -f $(OBJDIR)/*

realclean: clean
	-@rm -f $(UTILS_LIB) $(UTILS_LIB_SHARED)

library: ${COLAB_FILES_OBJS}
	ar rcs $(UTILS_LIB) \
	   $(COLAB_FILES_OBJS)
	$(CC) $(LDFLAGS) -o ${UTILS_LIB_SHARED} \
	   $(COLAB_FILES_OBJS)
