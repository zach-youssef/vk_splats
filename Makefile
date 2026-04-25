# Compiler
CC = /usr/bin/g++
CXX = $(CC)

# Homebrew Include Path + Project Headers
CFLAGS = -std=c++20 -Wc++11-extensions -I/opt/homebrew/include/opencv4 -I./include -I/usr/local/include -Wc++14-extensions -DENABLE_PRECOMPILED_HEADERS=OFF -I/Users/zyoussef/code/vulkan_test/vulkan_test -I/Users/zyoussef/stb/ -I/Users/zyoussef/VulkanSDK/1.4.321.0/macOS/include/
CXXFLAGS = $(CFLAGS)

# Homebrew Library Path + user/local
LDFLAGS = -L/opt/homebrew/lib -L/usr/local/lib

# Opencv Libraries
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_calib3d -lglfw -lvulkan -lopencv_viz

OUT = build
SRC = src

.PHONY: clean shaders

# Rule for building Cpp objects
$(OUT)/%.o : $(SRC)/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS) $(CLI_FLAGS)

# Rule for building compute shaders
spirv/%.spv : shaders/%.comp
	glslc $< -o $@ --target-env=vulkan1.4

_COMPUTE = preprocessSplats assignSplatsToTile single_radixsort rasterizeSplats rasterizeBackwards preprocessBackwards updateModel
COMPUTE = $(patsubst %,spirv/%.spv,$(_COMPUTE))

# Rules for our simple vertex & fragment shader
spirv/vert.spv : shaders/displayTraining.vert
	glslc $< -o $@ --target-env=vulkan1.4

spirv/frag.spv : shaders/displayTraining.frag
	glslc $< -o $@ --target-env=vulkan1.4

_GRAPHICS = vert frag
GRAPHICS = $(patsubst %,spirv/%.spv,$(_GRAPHICS))

shaders: $(COMPUTE) $(GRAPHICS)

# Application Rules
vk_train_splats: build/vk_train.o 
	$(CC) $^ -o $@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /usr/local/lib ./$@

clean:
	rm -f vk_train_splats build/* spirv/*