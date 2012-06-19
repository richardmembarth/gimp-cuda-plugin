#
# CUDA Makefile, based on http://www.eecs.berkeley.edu/~mjmurphy/makefile
#

# Location of the gimp plugin directory
INSTALL_PATH    = $(HOME)/.gimp-2.6/plug-ins

# Location of the CUDA SDK
CUDA_SDK_PATH       ?= /opt/cuda/sdk
# handle spaces in SDK filename
space = $(empty) $(empty)
# $(call space_to_question,file_name)
space_to_question = $(subst $(space),?,$1)
# $(call wildcard_spaces,file_name)
wildcard_spaces = $(wildcard $(call space_to_question,$(CUDA_SDK_PATH)/C))

ifeq "$(wildcard $(call space_to_question,$(CUDA_SDK_PATH)/C))" ""
    # CUDA 2.2 and below
    SDK_PATH        = "$(CUDA_SDK_PATH)"
    CUDA_LIB_EXT 	= lib
    LCUTIL          = -lcutil
else
    # CUDA 2.3 and above
    SDK_PATH        = "$(CUDA_SDK_PATH)/C"
    ifeq (x86_64, $(shell uname -m))
        CUDA_LIB_EXT 	= lib64
    else
        CUDA_LIB_EXT 	= lib
    endif
    ifeq "$(wildcard $(call space_to_question,$(CUDA_SDK_PATH)/C/lib/libcutil_*))" ""
        # CUDA 2.3
        LCUTIL          = -lcutil
    else
        # CUDA 3.0 and above
        ifeq (x86_64, $(shell uname -m))
            LCUTIL          = -lcutil_x86_64
        else
            LCUTIL          = -lcutil_i386
        endif
    endif
endif

# check if we have to build for 64bit on Snow Leopard
# works for default 64 bit MacPorts installation of GIMP
ifeq (Darwin,$(shell uname -s))
    BIT         = -m64
    LCUTIL      = -lcutil_x86_64
else
    BIT         =
endif


# This is where the cuda runtime libraries and includes reside
CUDAROOT  := /opt/cuda


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
CCLIB_PATHS := $(LINK_GIMP) -L$(SDK_PATH)/common/lib
CCINC_PATHS := $(INCLUDE_GIMP) -I$(SDK_PATH)/common/inc
CCLIBRARIES := $(LCUTIL)

#----- CUDA compilation options -----

NVCC        := $(CUDAROOT)/bin/nvcc
CUFLAGS     += $(BIT) $(OPT) $(DBG) -use_fast_math
CULIB_PATHS := -L$(CUDAROOT)/$(CUDA_LIB_EXT) -L$(SDK_PATH)/lib -L$(SDK_PATH)/common/lib
CUINC_PATHS := -I$(CUDAROOT)/include $(INCLUDE_GIMP) -I$(SDK_PATH)/common/inc
CULIBRARIES := -lcudart $(LCUTIL)

LIB_PATHS   := $(CULIB_PATHS) $(CCLIB_PATHS) $(CLIB_PATHS)
LIBRARIES   := $(CULIBRARIES) $(CCLIBRARIES) $(CLIBRARIES)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

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
	cp $(TARGET) $(INSTALL_PATH)/

