cmake_minimum_required(VERSION 3.13)  # CMake version check
project(simple_example LANGUAGES CXX CUDA)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 17)            # Enable c++14 standard
find_package(CUDA REQUIRED)

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES main.cpp
                 utilities/meshFunctions.cpp
    )


add_subdirectory(../libShapeDescriptor ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)

include_directories(
    "."
    ../libShapeDescriptor/src
    ../libShapeDescriptor/lib
    ../libShapeDescriptor/lib/nvidia-samples-common/
    ${CUDA_INCLUDE_DIRS}
)


# Add executable target with source files listed in SOURCE_FILES variable
add_executable(simple_example ${SOURCE_FILES})
target_link_libraries(simple_example PRIVATE ShapeDescriptor)

target_include_directories(
    simple_example PRIVATE 
    ../libShapeDescriptor/src
    ../libShapeDescriptor/lib
    ../libShapeDescriptor/lib/nvidia-samples-common/
    "${CMAKE_CURRENT_SOURCE_DIR}"
    )

