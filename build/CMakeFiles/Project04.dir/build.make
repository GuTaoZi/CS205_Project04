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
CMAKE_SOURCE_DIR = /mnt/e/Cpp/Project04

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/Cpp/Project04/build

# Include any dependencies generated for this target.
include CMakeFiles/Project04.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Project04.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Project04.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Project04.dir/flags.make

CMakeFiles/Project04.dir/matrix.c.o: CMakeFiles/Project04.dir/flags.make
CMakeFiles/Project04.dir/matrix.c.o: ../matrix.c
CMakeFiles/Project04.dir/matrix.c.o: CMakeFiles/Project04.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/Project04.dir/matrix.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Project04.dir/matrix.c.o -MF CMakeFiles/Project04.dir/matrix.c.o.d -o CMakeFiles/Project04.dir/matrix.c.o -c /mnt/e/Cpp/Project04/matrix.c

CMakeFiles/Project04.dir/matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Project04.dir/matrix.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/Cpp/Project04/matrix.c > CMakeFiles/Project04.dir/matrix.c.i

CMakeFiles/Project04.dir/matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Project04.dir/matrix.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/Cpp/Project04/matrix.c -o CMakeFiles/Project04.dir/matrix.c.s

CMakeFiles/Project04.dir/benchmark.c.o: CMakeFiles/Project04.dir/flags.make
CMakeFiles/Project04.dir/benchmark.c.o: ../benchmark.c
CMakeFiles/Project04.dir/benchmark.c.o: CMakeFiles/Project04.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/Project04.dir/benchmark.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Project04.dir/benchmark.c.o -MF CMakeFiles/Project04.dir/benchmark.c.o.d -o CMakeFiles/Project04.dir/benchmark.c.o -c /mnt/e/Cpp/Project04/benchmark.c

CMakeFiles/Project04.dir/benchmark.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Project04.dir/benchmark.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/Cpp/Project04/benchmark.c > CMakeFiles/Project04.dir/benchmark.c.i

CMakeFiles/Project04.dir/benchmark.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Project04.dir/benchmark.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/Cpp/Project04/benchmark.c -o CMakeFiles/Project04.dir/benchmark.c.s

CMakeFiles/Project04.dir/matmul.c.o: CMakeFiles/Project04.dir/flags.make
CMakeFiles/Project04.dir/matmul.c.o: ../matmul.c
CMakeFiles/Project04.dir/matmul.c.o: CMakeFiles/Project04.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/Project04.dir/matmul.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/Project04.dir/matmul.c.o -MF CMakeFiles/Project04.dir/matmul.c.o.d -o CMakeFiles/Project04.dir/matmul.c.o -c /mnt/e/Cpp/Project04/matmul.c

CMakeFiles/Project04.dir/matmul.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/Project04.dir/matmul.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/Cpp/Project04/matmul.c > CMakeFiles/Project04.dir/matmul.c.i

CMakeFiles/Project04.dir/matmul.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/Project04.dir/matmul.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/Cpp/Project04/matmul.c -o CMakeFiles/Project04.dir/matmul.c.s

# Object files for target Project04
Project04_OBJECTS = \
"CMakeFiles/Project04.dir/matrix.c.o" \
"CMakeFiles/Project04.dir/benchmark.c.o" \
"CMakeFiles/Project04.dir/matmul.c.o"

# External object files for target Project04
Project04_EXTERNAL_OBJECTS =

Project04: CMakeFiles/Project04.dir/matrix.c.o
Project04: CMakeFiles/Project04.dir/benchmark.c.o
Project04: CMakeFiles/Project04.dir/matmul.c.o
Project04: CMakeFiles/Project04.dir/build.make
Project04: /usr/local/lib/libopenblas.a
Project04: CMakeFiles/Project04.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable Project04"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Project04.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Project04.dir/build: Project04
.PHONY : CMakeFiles/Project04.dir/build

CMakeFiles/Project04.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Project04.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Project04.dir/clean

CMakeFiles/Project04.dir/depend:
	cd /mnt/e/Cpp/Project04/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/Cpp/Project04 /mnt/e/Cpp/Project04 /mnt/e/Cpp/Project04/build /mnt/e/Cpp/Project04/build /mnt/e/Cpp/Project04/build/CMakeFiles/Project04.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Project04.dir/depend

