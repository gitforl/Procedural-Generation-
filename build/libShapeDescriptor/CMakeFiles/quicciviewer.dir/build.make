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
include libShapeDescriptor/CMakeFiles/quicciviewer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include libShapeDescriptor/CMakeFiles/quicciviewer.dir/compiler_depend.make

# Include the progress variables for this target.
include libShapeDescriptor/CMakeFiles/quicciviewer.dir/progress.make

# Include the compile flags for this target's objects.
include libShapeDescriptor/CMakeFiles/quicciviewer.dir/flags.make

libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o: libShapeDescriptor/CMakeFiles/quicciviewer.dir/flags.make
libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o: /lhome/lukashg/libShapeDescriptor/tools/quicciviewer/main.cpp
libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o: libShapeDescriptor/CMakeFiles/quicciviewer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/lhome/lukashg/Procedural-Generation-/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o"
	cd /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o -MF CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o.d -o CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o -c /lhome/lukashg/libShapeDescriptor/tools/quicciviewer/main.cpp

libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.i"
	cd /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /lhome/lukashg/libShapeDescriptor/tools/quicciviewer/main.cpp > CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.i

libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.s"
	cd /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /lhome/lukashg/libShapeDescriptor/tools/quicciviewer/main.cpp -o CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.s

# Object files for target quicciviewer
quicciviewer_OBJECTS = \
"CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o"

# External object files for target quicciviewer
quicciviewer_EXTERNAL_OBJECTS =

libShapeDescriptor/quicciviewer: libShapeDescriptor/CMakeFiles/quicciviewer.dir/tools/quicciviewer/main.cpp.o
libShapeDescriptor/quicciviewer: libShapeDescriptor/CMakeFiles/quicciviewer.dir/build.make
libShapeDescriptor/quicciviewer: libShapeDescriptor/libShapeDescriptor.a
libShapeDescriptor/quicciviewer: /usr/local/cuda/lib64/libcudart_static.a
libShapeDescriptor/quicciviewer: /usr/lib/x86_64-linux-gnu/librt.a
libShapeDescriptor/quicciviewer: libShapeDescriptor/fast-lzma2/libfast-lzma2.a
libShapeDescriptor/quicciviewer: libShapeDescriptor/CMakeFiles/quicciviewer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/lhome/lukashg/Procedural-Generation-/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable quicciviewer"
	cd /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quicciviewer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libShapeDescriptor/CMakeFiles/quicciviewer.dir/build: libShapeDescriptor/quicciviewer
.PHONY : libShapeDescriptor/CMakeFiles/quicciviewer.dir/build

libShapeDescriptor/CMakeFiles/quicciviewer.dir/clean:
	cd /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor && $(CMAKE_COMMAND) -P CMakeFiles/quicciviewer.dir/cmake_clean.cmake
.PHONY : libShapeDescriptor/CMakeFiles/quicciviewer.dir/clean

libShapeDescriptor/CMakeFiles/quicciviewer.dir/depend:
	cd /lhome/lukashg/Procedural-Generation-/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /lhome/lukashg/Procedural-Generation- /lhome/lukashg/libShapeDescriptor /lhome/lukashg/Procedural-Generation-/build /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor /lhome/lukashg/Procedural-Generation-/build/libShapeDescriptor/CMakeFiles/quicciviewer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libShapeDescriptor/CMakeFiles/quicciviewer.dir/depend

