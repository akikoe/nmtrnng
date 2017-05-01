CXX=g++

SHARE_LOCATION=./N3LP
EIGEN_LOCATION=pathToEigen # TODO: Modify the path to Eigen
BUILD_DIR=objs

CXXFLAGS =-Wall
CXXFLAGS+=-O3
CXXFLAGS+=-std=c++11
CXXFLAGS+=-mavx
CXXFLAGS+=-DUSE_FLOAT # Comment out when running gradient checking
CXXFLAGS+=-DUSE_EIGEN_TANH
CXXFLAGS+=-lm
CXXFLAGS+=-fomit-frame-pointer
CXXFLAGS+=-fno-schedule-insns2
CXXFLAGS+=-fexceptions
CXXFLAGS+=-funroll-loops
CXXFLAGS+=-march=native
CXXFLAGS+=-m64
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
CXXFLAGS+=-DEIGEN_NO_DEBUG
CXXFLAGS+=-DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+=-I$(EIGEN_LOCATION)
CXXFLAGS+=-I$(BOOST_LOCATION)
CXXFLAGS+=-I$(SHARE_LOCATION)
CXXFLAGS+=-fopenmp

SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

PROGRAM=nmtrnng

all : $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(PROGRAM))

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/$(PROGRAM) : $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(CXX) $(CXXFLAGS) $(CXXLIBS) -o $@ $^
	mv $(BUILD_DIR)/$(PROGRAM) ./
	rm -f ?*~

clean:
	rm -f $(BUILD_DIR)/* $(PROGRAM) ?*~
