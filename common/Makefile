OBJDIR= ./obj
SRCDIR= ./src
INCDIR= ./include
CC=g++
CFLAGS=-I$(INCDIR) -Wall -lgmp -std=c++11 -Wno-deprecated-declarations # -pedantic

RELEASE?="" # Potentially set by make_XXX or the user from the command line

# Check if release mode
ifeq ($(RELEASE), -release)
   # No debugging
   # Yes optimization
   CFLAGS += -O3
else
   # Yes debugging
   CFLAGS += -ggdb -g
   # Minimal optimization
   CFLAGS += -Og
endif

.DEFAULT_GOAL := final

SRCS=$(foreach dir, $(SRCDIR), $(wildcard $(dir)/*.C))
TMPS=$(foreach dir, $(SRCDIR), $(patsubst $(dir)/%, $(OBJDIR)/%, $(wildcard $(dir)/*.C)))
OBJS=$(patsubst %.C, %.o, $(TMPS))
#OBJS=$(patsubst $($(PURE_SRCS:.C=.o )

vpath %.C $(SRCDIR)
vpath %.h $(INCDIR)

$(OBJS): | $(OBJDIR)

$(OBJDIR): 
	mkdir -p $(OBJDIR)

final: $(OBJS)
#final: 
	echo $(OBJS)

$(OBJDIR)/%.o: %.C 
	$(CC) $(CFLAGS) -c $< -o $@

clean: realclean

realclean:
	-rm -f $(OBJDIR)/*
