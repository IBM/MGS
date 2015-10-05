CFLAGS= -g -Wall -O3                                                            
OBJDIR= ./obj

SRCS=$(wildcard *.C)
OBJS=$(SRCS:.C=.o )

$(OBJS): | $(OBJDIR)

$(OBJDIR): 
	mkdir -p $(OBJDIR)
	
all:$(OBJS)

%.o: %.C 
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)/$@

