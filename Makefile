CC = gcc
CXX = g++
CFLAGS = -std=c99 -ggdb -I../common
LIBS = -lm -lOpenCL

INCLUDES = imagenet_labels.h

all: vgg0 vgg1

vgg0: main0.o
	$(CC) main0.o ../common/libutils.a -o vgg0 $(LIBS)

main0.o: main0.c $(INCLUDES)
	$(CC) -c $(CFLAGS) main0.c

vgg1: main1.o
	$(CC) main1.o ../common/libutils.a -o vgg1 $(LIBS)

main1.o: main1.c $(INCLUDES)
	$(CC) -c $(CFLAGS) main1.c



clean:
	rm *.o vgg
