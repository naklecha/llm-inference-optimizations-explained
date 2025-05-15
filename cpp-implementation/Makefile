CXX       = g++
CXXFLAGS  = -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
            -DTORCH_ENABLE_FASTER_RCNN -DUSE_CUDA \
            -D_GLIBCXX_USE_CXX11_ABI=1

# ─── adjust this to the folder that contains "lib/" and "include/"
TORCH_DIR = /home/naklecha/libtorch

INCLUDES  =  -I$(TORCH_DIR)/include \
             -I$(TORCH_DIR)/include/torch/csrc/api/include \
             -I./cnpy

# --- common libs
COMMON_LIBS = -lc10 -lz -pthread -ldl

# --- if the CUDA library exists, link it with --no-as-needed
ifeq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    TORCH_LIBS = -ltorch -ltorch_cpu $(COMMON_LIBS)
else
    TORCH_LIBS = -Wl,--no-as-needed -ltorch_cuda -ltorch -ltorch_cpu $(COMMON_LIBS)
endif

RPATH     = -Wl,-rpath,$(TORCH_DIR)/lib
LDIRS     = -L$(TORCH_DIR)/lib        # <── search path must precede -l flags

SRC       = 4-in-cpp.cpp cnpy/cnpy.cpp
TARGET    = app

all: $(TARGET)

$(TARGET): $(SRC)
	@set -e; \
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $@ $(LDIRS) $(TORCH_LIBS) $(RPATH) 2> build_errors.log || ( \
		awk '{print NR ": " $$0}' build_errors.log 1>&2; \
		exit 1 \
	)

clean:
	rm -f $(TARGET)
