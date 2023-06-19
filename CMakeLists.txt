cmake_minimum_required(VERSION 3.13)  # CMake version check
# set(CMAKE_CUDA_ARCHITECTURES 70)
project(simple_example LANGUAGES CXX CUDA)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 17)            # Enable c++14 standard
find_package(CUDA REQUIRED)

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES main.cpp
                 utilities/meshFunctions.cpp
                 utilities/boundingBox.cpp
                 utilities/cgalMeshFunctions.cpp
                 utilities/mathUtilities.cpp
                 utilities/generalUtilities.cpp
                 utilities/descriptorDistance.cpp
                 utilities/descriptorDistance.cu
                 descriptors/BaseDescriptor.cpp
                 descriptors/QUICCIDescriptor.cpp
                 descriptors/RICIDescriptorTester.cpp
                 descriptors/SIDescriptorTester.cpp
                 descriptors/FPFHDescriptorTester.cpp
                 meshModifier/meshModifier.cpp
                 meshModifier/model.cpp
                 meshModifier/cgalMesh.cpp
                 openglHandler/openglHandler.cpp
                 openglHandler/openglMesh.cpp
                 lib/glad/src/glad.cpp
                 distances/quicciDistances.cu
                 distances/riciDistances.cu
                 distances/siDistances.cu
                 distances/utility.cu
    )

#
# GLFW options
#
option (GLFW_INSTALL OFF)
option (GLFW_BUILD_DOCS OFF)
option (GLFW_BUILD_EXAMPLES OFF)
option (GLFW_BUILD_TESTS OFF)
add_subdirectory (lib/glfw)


add_subdirectory(../libShapeDescriptor ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)

include_directories(
    "."
    ../libShapeDescriptor/src
    ../libShapeDescriptor/lib
    ../libShapeDescriptor/lib/nvidia-samples-common/
    ${CUDA_INCLUDE_DIRS}
)


include_directories (
    lib/
    lib/glm/
    lib/glad/include/
    lib/glfw/include/
)

find_package(CGAL)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(simple_example ${SOURCE_FILES})
target_link_libraries(simple_example PRIVATE ShapeDescriptor glfw)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(simple_example PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CGAL_FOUND)
    target_link_libraries(simple_example PUBLIC CGAL::CGAL)
endif()

target_include_directories(
    simple_example PRIVATE 
    ../libShapeDescriptor/src
    ../libShapeDescriptor/lib
    ../libShapeDescriptor/lib/nvidia-samples-common/
    "${CMAKE_CURRENT_SOURCE_DIR}"
    )

