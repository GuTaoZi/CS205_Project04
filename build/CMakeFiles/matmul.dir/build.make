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
include CMakeFiles/matmul.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matmul.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matmul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matmul.dir/flags.make

CMakeFiles/matmul.dir/src/benchmark.c.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/src/benchmark.c.o: ../src/benchmark.c
CMakeFiles/matmul.dir/src/benchmark.c.o: CMakeFiles/matmul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/matmul.dir/src/benchmark.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/matmul.dir/src/benchmark.c.o -MF CMakeFiles/matmul.dir/src/benchmark.c.o.d -o CMakeFiles/matmul.dir/src/benchmark.c.o -c /mnt/e/Cpp/Project04/src/benchmark.c

CMakeFiles/matmul.dir/src/benchmark.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/matmul.dir/src/benchmark.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/Cpp/Project04/src/benchmark.c > CMakeFiles/matmul.dir/src/benchmark.c.i

CMakeFiles/matmul.dir/src/benchmark.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/matmul.dir/src/benchmark.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/Cpp/Project04/src/benchmark.c -o CMakeFiles/matmul.dir/src/benchmark.c.s

CMakeFiles/matmul.dir/src/matmul.c.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/src/matmul.c.o: ../src/matmul.c
CMakeFiles/matmul.dir/src/matmul.c.o: CMakeFiles/matmul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/matmul.dir/src/matmul.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/matmul.dir/src/matmul.c.o -MF CMakeFiles/matmul.dir/src/matmul.c.o.d -o CMakeFiles/matmul.dir/src/matmul.c.o -c /mnt/e/Cpp/Project04/src/matmul.c

CMakeFiles/matmul.dir/src/matmul.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/matmul.dir/src/matmul.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/Cpp/Project04/src/matmul.c > CMakeFiles/matmul.dir/src/matmul.c.i

CMakeFiles/matmul.dir/src/matmul.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/matmul.dir/src/matmul.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/Cpp/Project04/src/matmul.c -o CMakeFiles/matmul.dir/src/matmul.c.s

CMakeFiles/matmul.dir/src/matrix.c.o: CMakeFiles/matmul.dir/flags.make
CMakeFiles/matmul.dir/src/matrix.c.o: ../src/matrix.c
CMakeFiles/matmul.dir/src/matrix.c.o: CMakeFiles/matmul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/matmul.dir/src/matrix.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/matmul.dir/src/matrix.c.o -MF CMakeFiles/matmul.dir/src/matrix.c.o.d -o CMakeFiles/matmul.dir/src/matrix.c.o -c /mnt/e/Cpp/Project04/src/matrix.c

CMakeFiles/matmul.dir/src/matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/matmul.dir/src/matrix.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/e/Cpp/Project04/src/matrix.c > CMakeFiles/matmul.dir/src/matrix.c.i

CMakeFiles/matmul.dir/src/matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/matmul.dir/src/matrix.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/e/Cpp/Project04/src/matrix.c -o CMakeFiles/matmul.dir/src/matrix.c.s

# Object files for target matmul
matmul_OBJECTS = \
"CMakeFiles/matmul.dir/src/benchmark.c.o" \
"CMakeFiles/matmul.dir/src/matmul.c.o" \
"CMakeFiles/matmul.dir/src/matrix.c.o"

# External object files for target matmul
matmul_EXTERNAL_OBJECTS =

matmul: CMakeFiles/matmul.dir/src/benchmark.c.o
matmul: CMakeFiles/matmul.dir/src/matmul.c.o
matmul: CMakeFiles/matmul.dir/src/matrix.c.o
matmul: CMakeFiles/matmul.dir/build.make
matmul: /usr/local/lib/libopenblas.a
matmul: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
matmul: /usr/lib/x86_64-linux-gnu/libpthread.a
matmul: CMakeFiles/matmul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/Cpp/Project04/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable matmul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matmul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matmul.dir/build: matmul
.PHONY : CMakeFiles/matmul.dir/build

CMakeFiles/matmul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matmul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matmul.dir/clean

CMakeFiles/matmul.dir/depend:
	cd /mnt/e/Cpp/Project04/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/Cpp/Project04 /mnt/e/Cpp/Project04 /mnt/e/Cpp/Project04/build /mnt/e/Cpp/Project04/build /mnt/e/Cpp/Project04/build/CMakeFiles/matmul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matmul.dir/depend
