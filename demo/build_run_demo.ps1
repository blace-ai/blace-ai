$env:BLACE_AI_CMAKE_DIR = "../cmake"
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
.\build\Release\demo.exe