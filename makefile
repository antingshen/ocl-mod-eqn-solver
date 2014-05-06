OBJS=ocl-solver.o clhelp.o
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
OCL_INC=/opt/opencl-headers/include
OCL_LIB=/opt/opencl-headers/include

%.o: %.cpp clhelp.h
	g++ -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	g++ ocl-solver.o clhelp.o -o ocl-solver -L$(OCL_LIB) -lOpenCL
	./ocl-solver
endif

clean:
	rm -rf ocl-solver $(OBJS)