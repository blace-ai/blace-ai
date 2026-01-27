export BLACE_AI_CMAKE_DIR="../cmake"
cmake -S . -B build
cmake --build build --config Release
./build/demo