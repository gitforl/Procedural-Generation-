# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /lhome/lukashg/Procedural-Generation-

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /lhome/lukashg/Procedural-Generation-/build

# Include any dependencies generated for this target.
include CMakeFiles/simple_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/simple_example.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/simple_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/simple_example.dir/flags.make

CMakeFiles/simple_example.dir/main.cpp.o: CMakeFiles/simple_example.dir/flags.make
CMakeFiles/simple_example.dir/main.cpp.o: ../main.cpp
CMakeFiles/simple_example.dir/main.cpp.o: CMakeFiles/simple_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/lhome/lukashg/Procedural-Generation-/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/simple_example.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/simple_example.dir/main.cpp.o -MF CMakeFiles/simple_example.dir/main.cpp.o.d -o CMakeFiles/simple_example.dir/main.cpp.o -c /lhome/lukashg/Procedural-Generation-/main.cpp

CMakeFiles/simple_example.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple_example.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /lhome/lukashg/Procedural-Generation-/main.cpp > CMakeFiles/simple_example.dir/main.cpp.i

CMakeFiles/simple_example.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple_example.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /lhome/lukashg/Procedural-Generation-/main.cpp -o CMakeFiles/simple_example.dir/main.cpp.s

# Object files for target simple_example
simple_example_OBJECTS = \
"CMakeFiles/simple_example.dir/main.cpp.o"

# External object files for target simple_example
simple_example_EXTERNAL_OBJECTS =

simple_example: CMakeFiles/simple_example.dir/main.cpp.o
simple_example: CMakeFiles/simple_example.dir/build.make
simple_example: libShapeDescriptor/libShapeDescriptor.a
simple_example: /usr/local/cuda/lib64/libcudart_static.a
simple_example: /usr/lib/x86_64-linux-gnu/librt.a
simple_example: libShapeDescriptor/fast-lzma2/libfast-lzma2.a
simple_example: CMakeFiles/simple_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/lhome/lukashg/Procedural-Generation-/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable simple_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/simple_example.dir/build: simple_example
.PHONY : CMakeFiles/simple_example.dir/build

CMakeFiles/simple_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/simple_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/simple_example.dir/clean

CMakeFiles/simple_example.dir/depend:
	cd /lhome/lukashg/Procedural-Generation-/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /lhome/lukashg/Procedural-Generation- /lhome/lukashg/Procedural-Generation- /lhome/lukashg/Procedural-Generation-/build /lhome/lukashg/Procedural-Generation-/build /lhome/lukashg/Procedural-Generation-/build/CMakeFiles/simple_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/simple_example.dir/depend

