# Procedural-Generation-

This repository contains the implementation of the system produced during the master thesis.


As different solutions were explored during the development, the current repository contains many files that no longer serve a functional purpose
beyond demonstrating the path taken to reach the project's final state. Therefore, to make navigation easier, and make clear what I developed myself an overview is presented here.

#### Folder Strucuture with explanation

┝ descriptors: No longer in use\
┝ distances: Implementation of distance functions for different Shape Descriptors\
┝ images: Output for created images\
┝ lib: External libraries\
┝ meshModifier: Misleading name, contains Model, encapsulates three mesh represenations employed in different applications\
┝ objects: Folder with 3D polygon meshes employed thoughout the system. When saving altered version, they end up here as well\
┝ openglHandler: The class used to work with OpenGL, and some related minor classes\
┝ res/shaders: Shaders used by OpenGL to render properly\
┝ utilities: A folder containing varying utility functionality\
┝ CMakeLists.txt: CMake file used to properly configure compilation\
┗ main.cpp: Main Executable, also contains System class implementation and classes that proivde interaction with several descriptor types


#### NOTE:
  The lib folder contains external code needed to run OpenGL and was not developed by me. It was sourced from Barts repository found on https://github.com/bartvbl/Dissimilarity-Tree-Reproduction. The descriptor folder contains a previous solution attemping to provide a standarized interface to different shape descriptors, developed during the project done the previous semester, but these are no longer in use. Lastly, several functions found in utilites/meshFunctions (FindSimilarVerticesIndices, MoveVertexAlongNormal, MapVertexIndices, VertexToAverageNormalMap, MoveVerticesAlongAverageNormal) were also developed then as well, however, these are still in use for generating noise.

  The distance functions for seveal descriptor types found in the distances folder are modified from implementations by Bart found in https://github.com/bartvbl/libShapeDescriptor. These were extended to provide several distance types as discussed in the thesis, and modified to better work the system. Several of the functionalities found in openglHandler/openglHandler were also inspired by others. Much of the code was created following examples found on https://learnopengl.com/ including shader.cpp found in the folder above. CreateMeshWithOcclusion and some of its subfunctions were inspired by the solution found on https://github.com/bartvbl/Dissimilarity-Tree-Reproduction/blob/master/src/partialRetrieval/tools/querysetgenerator/main.cpp.

#### Main:
  Contains the executable, as well the implementation of the overarching system. In the executable, this system is first instantiated, then configured to generate the desired synthetic data, descriptors and distances. It can also be configured to visualize the desired object. Lastly the system is run, where it follows the same previous sequence. It generates the data, then visualizes it, before finally computing the specified distances. The produced distance distribution is approximated using a histogram, stored in a json as an array of integers.

##### ApplyDisturbances
  As the system consolidates the different functionalty it also clearly show which concrete functions are used to perform particular tasks. For example, the process for generating synthetic data is done in ApplyDisturbances, where each is applied consecutively within their own scoped block.

  ###### ApplyClutter:
  Generates a cluttered scene. Does so using functionality which relies on utilities/boundingBox 
  
  ###### ApplyOcclusion:
  Alters the scene to exhibit occlusion. Does so using functionality provided by OpenGLHandler.
  
  ###### ApplyNoise:
  Applies noise using functionality from utilities/meshFunctions

##### Draw
  Uses the functionality from OpenGLHandler and the current system configuration to visualize the desired object(s).
##### TestDistances
  Uses the current system configuration to decide both which descriptor and distance type to test. The functionality needed to work with the descriptors are implemented in classes deriving from a general abstract class. This abstract class therefore provides uniform access to the different types.
  
