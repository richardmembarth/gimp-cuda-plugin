#
# CUDA Makefile, based on http://www.eecs.berkeley.edu/~mjmurphy/makefile
#

# This is where the CUDA runtime libraries and includes reside
CUDAROOT  := /opt/cuda

# Set variables depending on platform
ifeq (Darwin,$(shell uname -s))
    BIT             = -m64
    CUDA_LIB_EXT    = lib
    # Location of the gimp plugin directory
    INSTALL_PATH    = $(HOME)/Library/GIMP/2.8/plug-ins
else
    CUDA_LIB_EXT    = lib64
    # Location of the gimp plugin directory
    INSTALL_PATH    = $(HOME)/.gimp-2.8/plug-ins
    BIT             =
endif


# Fill in the name of the output binary here
TARGET    := multi_res_filter

# List of sources, with .c, .cu, and .cpp extensions
SOURCES := \
    gimp_gui.cpp \
    gimp_main.cpp \
    multi_res_cpu.cpp \
    multi_res_host.cu


# Other things that need to be built, e.g. .cubin files
EXTRADEPS := \
    Makefile \
    multi_res_device.cu \
    multi_res_host.hpp \
    defines_cpu.hpp \
    defines_gpu.hpp \
    gimp_gui.hpp \
    gimp_main.hpp \
    multi_res_cpu.hpp


# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

OPT  ?= -O3
DBG  ?= 
WARN ?= -Wall

# flags required to link and compile against gimp
INCLUDE_GIMP_pt = $(shell pkg-config --cflags gimpui-2.0)
INCLUDE_GIMP    = $(subst -pthread,,$(INCLUDE_GIMP_pt))
LINK_GIMP       = $(shell pkg-config --libs gimpui-2.0)

#----- C compilation options ------
GCC        := /usr/bin/gcc
CFLAGS     += $(BIT) $(OPT) $(DBG) $(WARN)
CLIB_PATHS :=

CINC_PATHS := -I $(CUDAROOT)/include
CLIBRARIES := -lcudart


#----- C++ compilation options ------
GPP         := /usr/bin/g++
CCFLAGS     += $(BIT) $(OPT) $(DBG) $(WARN)
CCLIB_PATHS := $(LINK_GIMP)
CCINC_PATHS := $(INCLUDE_GIMP)

#----- CUDA compilation options -----

NVCC        := $(CUDAROOT)/bin/nvcc
CUFLAGS     += $(BIT) $(OPT) $(DBG) -use_fast_math
CULIB_PATHS := -L$(CUDAROOT)/$(CUDA_LIB_EXT)
CUINC_PATHS := -I$(CUDAROOT)/include $(INCLUDE_GIMP)
CULIBRARIES := -lcudart $(LCUTIL)

LIB_PATHS   := $(CULIB_PATHS) $(CCLIB_PATHS) $(CLIB_PATHS)
LIBRARIES   := $(CULIBRARIES) $(CLIBRARIES)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and CUDA files.

CSOURCES  := $(filter %.c ,$(SOURCES))
CCSOURCES := $(filter %.cpp,$(SOURCES))
CUSOURCES := $(filter %.cu,$(SOURCES))

# This code generates a list of object files by replacing filename extensions

OBJECTS := $(patsubst %.c,%.o ,$(CSOURCES))  \
           $(patsubst %.cu,%.o,$(CUSOURCES)) \
           $(patsubst %.cpp,%.o,$(CCSOURCES))


#----- Build rules ------

$(TARGET): $(EXTRADEPS) 


$(TARGET): $(OBJECTS)
	$(GPP) -fPIC $(BIT) -o $@ $(OBJECTS) $(LIB_PATHS) $(LIBRARIES)

%.o: %.cu
	$(NVCC) -c $^ $(CUFLAGS) $(CUINC_PATHS) -o $@ 

%.cubin: %.cu
	$(NVCC) -cubin $(CUFLAGS) $(CUINC_PATHS) $^

%.o: %.cpp
	$(GPP) -c $^ $(CCFLAGS) $(CCINC_PATHS) -o $@

%.o: %.c
	$(GCC) -c $^ $(CFLAGS) $(CINC_PATHS) -o $@

clean:
	rm -f *.o $(TARGET) *.linkinfo

install: $(TARGET)
	mkdir -p $(INSTALL_PATH)
	cp $(TARGET) $(INSTALL_PATH)

