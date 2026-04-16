cmake_minimum_required(VERSION 3.21)

project(BlaceDemo VERSION 1.0.0 LANGUAGES CXX)

option(BLACE_AI_USE_IPC "Use IPC version" OFF)

# Require BLACE_CMAKE_DIR env var
if(NOT DEFINED ENV{BLACE_AI_CMAKE_DIR})
    message(FATAL_ERROR "Environment variable BLACE_AI_CMAKE_DIR is not set. Please export it to the path of the blace.ai cmake folder.")
endif()
set(BLACE_AI_CMAKE_DIR $ENV{BLACE_AI_CMAKE_DIR})

# Use the path
if(BLACE_AI_USE_IPC)
    include("${BLACE_AI_CMAKE_DIR}/FindBlaceIPC.cmake")
else()
    include("${BLACE_AI_CMAKE_DIR}/FindBlace.cmake")
endif()

set(DEMO_SOURCES
    main.cpp
    RunDepthModel.h
    RunDepthModel.cpp
)

set(CMAKE_CXX_STANDARD 17)
add_executable(demo ${DEMO_SOURCES})

# link with blaceai library. if we are using ipc version, we need to manually link with opencv as BlaceAIIPC has no 3rdparty dependencies
if(BLACE_AI_USE_IPC)
    target_link_libraries(demo PUBLIC 3rdparty::BlaceAIIPC 3rdparty::BlaceAIOpenCV)
else()
    target_link_libraries(demo PUBLIC 3rdparty::BlaceAI)
endif()

message(DEBUG "Runtime files: ${BLACE_AI_RUNTIME_FILES}")

# copy shared libraries / executables to target folder
foreach(RUNTIME_FILE ${BLACE_AI_RUNTIME_FILES})
    add_custom_command(TARGET demo POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${RUNTIME_FILE}" $<TARGET_FILE_DIR:demo>
    )
endforeach()

# copy needed test data (input image, model payloads) to executable folder
add_custom_command(
  TARGET demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/test_butterfly.jpg $<TARGET_FILE_DIR:demo>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/depth_result_ref.png $<TARGET_FILE_DIR:demo>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/596060abd5d12ffd3e6e71ec561fed6f.bin $<TARGET_FILE_DIR:demo>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/ef5455e142d4fdb3c98a657607802db7.bin $<TARGET_FILE_DIR:demo>
)

# on apple we need to apply two changes to the final exe
if(APPLE)
    add_custom_command(TARGET demo
        POST_BUILD
        COMMAND install_name_tool -change @rpath/libopencv_world.406.dylib @rpath/libopencv_world.4.6.0.dylib $<TARGET_FILE:demo>
		COMMAND install_name_tool -add_rpath "@loader_path" $<TARGET_FILE:demo>
    )
endif()
