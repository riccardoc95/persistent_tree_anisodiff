cmake_minimum_required(VERSION 3.0)
project(pixhom)

# Add a shared library target
add_library(pixhom SHARED "pixhomology/pixhom.c")
add_library(graphom SHARED "pixhomology/exp/graphom.c")

# Specify include directories for this target
target_include_directories(pixhom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(graphom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Optionally, set the output directory for the DLL
#set_target_properties(pixhom PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Optionally, you can change the name of the output DLL
# set_target_properties(pixhom PROPERTIES OUTPUT_NAME pixhom)

# Optionally, if you want to control the export/import behavior
# of symbols, you can use the following line
# target_compile_definitions(pixhom PRIVATE MYLIB_EXPORTS)

