cmake_minimum_required(VERSION 3.21)

project(BlaceDemo VERSION 1.0.0 LANGUAGES CXX)

# include blaceai cmake setup.
include("../cmake/FindBlace.cmake")

set(DEMO_SOURCES
    main.cpp
    RunDepthModel.h
    RunDepthModel.cpp
)

set(CMAKE_CXX_STANDARD 17)
add_executable(demo ${DEMO_SOURCES})

# link with blaceai library
target_link_libraries(demo PUBLIC 3rdparty::BlaceAI)

# copy shared libraries to target folder
foreach(DLL_FILE ${BLACE_AI_COPY_LIBS})
    add_custom_command(TARGET demo POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy "${DLL_FILE}" $<TARGET_FILE_DIR:demo>
    )
endforeach()

# copy needed test data to executable folder
add_custom_command(
  TARGET demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/test_butterfly.jpg $<TARGET_FILE_DIR:demo>
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/data/9c4ad2d2e7daedc4c5710ff909c165b3.bin $<TARGET_FILE_DIR:demo>
)

# on apple we need to apply two changes to the final exe
if(APPLE)
    add_custom_command(TARGET demo
        POST_BUILD
        COMMAND install_name_tool -change @rpath/libopencv_world.406.dylib @rpath/libopencv_world.4.6.0.dylib $<TARGET_FILE:demo>
		COMMAND install_name_tool -add_rpath "@loader_path" $<TARGET_FILE:demo>
    )
endif()
