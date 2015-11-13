CFLAGS= -g -Wall -O3                                                            
OBJDIR= ./obj
SRCDIR= ./src
INCDIR= ./include

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
	echo $(OBJS)

$(OBJDIR)/%.o: %.C 
	$(CC) $(CFLAGS) -c $< -o $@

realclean:
	-rm $(OBJDIR)/*
