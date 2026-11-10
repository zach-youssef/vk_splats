LIBS_splat_view = -lglfw
# The PLY opener (splat_view.cpp) includes lib/native/'s reader and encoder
$(OUT)/splat_view/splat_view.o: $(wildcard lib/native/*.h)
