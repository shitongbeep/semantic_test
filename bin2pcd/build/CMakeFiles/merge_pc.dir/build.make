# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shitong/segementation/semantic_test/bin2pcd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shitong/segementation/semantic_test/bin2pcd/build

# Include any dependencies generated for this target.
include CMakeFiles/merge_pc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/merge_pc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/merge_pc.dir/flags.make

CMakeFiles/merge_pc.dir/merge_pc.cpp.o: CMakeFiles/merge_pc.dir/flags.make
CMakeFiles/merge_pc.dir/merge_pc.cpp.o: ../merge_pc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shitong/segementation/semantic_test/bin2pcd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/merge_pc.dir/merge_pc.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/merge_pc.dir/merge_pc.cpp.o -c /home/shitong/segementation/semantic_test/bin2pcd/merge_pc.cpp

CMakeFiles/merge_pc.dir/merge_pc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/merge_pc.dir/merge_pc.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shitong/segementation/semantic_test/bin2pcd/merge_pc.cpp > CMakeFiles/merge_pc.dir/merge_pc.cpp.i

CMakeFiles/merge_pc.dir/merge_pc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/merge_pc.dir/merge_pc.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shitong/segementation/semantic_test/bin2pcd/merge_pc.cpp -o CMakeFiles/merge_pc.dir/merge_pc.cpp.s

# Object files for target merge_pc
merge_pc_OBJECTS = \
"CMakeFiles/merge_pc.dir/merge_pc.cpp.o"

# External object files for target merge_pc
merge_pc_EXTERNAL_OBJECTS =

merge_pc: CMakeFiles/merge_pc.dir/merge_pc.cpp.o
merge_pc: CMakeFiles/merge_pc.dir/build.make
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_people.so
merge_pc: /usr/lib/x86_64-linux-gnu/libboost_system.so
merge_pc: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
merge_pc: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
merge_pc: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
merge_pc: /usr/lib/x86_64-linux-gnu/libboost_regex.so
merge_pc: /usr/lib/x86_64-linux-gnu/libqhull.so
merge_pc: /usr/lib/libOpenNI.so
merge_pc: /usr/lib/libOpenNI2.so
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libfreetype.so
merge_pc: /usr/lib/x86_64-linux-gnu/libz.so
merge_pc: /usr/lib/x86_64-linux-gnu/libjpeg.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpng.so
merge_pc: /usr/lib/x86_64-linux-gnu/libtiff.so
merge_pc: /usr/lib/x86_64-linux-gnu/libexpat.so
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
merge_pc: /usr/local/lib/libyaml-cpp.a
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_features.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_search.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_io.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
merge_pc: /usr/lib/x86_64-linux-gnu/libpcl_common.so
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libfreetype.so
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
merge_pc: /usr/lib/x86_64-linux-gnu/libz.so
merge_pc: /usr/lib/x86_64-linux-gnu/libGLEW.so
merge_pc: /usr/lib/x86_64-linux-gnu/libSM.so
merge_pc: /usr/lib/x86_64-linux-gnu/libICE.so
merge_pc: /usr/lib/x86_64-linux-gnu/libX11.so
merge_pc: /usr/lib/x86_64-linux-gnu/libXext.so
merge_pc: /usr/lib/x86_64-linux-gnu/libXt.so
merge_pc: CMakeFiles/merge_pc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shitong/segementation/semantic_test/bin2pcd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable merge_pc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/merge_pc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/merge_pc.dir/build: merge_pc

.PHONY : CMakeFiles/merge_pc.dir/build

CMakeFiles/merge_pc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/merge_pc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/merge_pc.dir/clean

CMakeFiles/merge_pc.dir/depend:
	cd /home/shitong/segementation/semantic_test/bin2pcd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shitong/segementation/semantic_test/bin2pcd /home/shitong/segementation/semantic_test/bin2pcd /home/shitong/segementation/semantic_test/bin2pcd/build /home/shitong/segementation/semantic_test/bin2pcd/build /home/shitong/segementation/semantic_test/bin2pcd/build/CMakeFiles/merge_pc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/merge_pc.dir/depend

