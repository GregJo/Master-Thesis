
# 
# External dependencies
# 

# find_package(THIRDPARTY REQUIRED)


# 
# Library name and options
# 

# Target name
set(target sutil_sdk)

# Exit here if required dependencies are not met
message(STATUS "Lib ${target}")

# Set API export file and macro
string(MAKE_C_IDENTIFIER ${target} target_id)
string(TOUPPER ${target_id} target_id)
set(feature_file         "include/${target}/${target}_features.h")
set(export_file          "include/${target}/${target}_export.h")
set(template_export_file "include/${target}/${target}_api.h")
set(export_macro         "${target_id}_API")


#include(FindOptiX)
#
## 
## Sources
## 
#
##set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
#set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")
#
##set(headers
##)
#
#set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}")
#set(OptiX_SDK_DIR "${OptiX_INSTALL_DIR}/SDK")
#set(SAMPLES_CUDA_DIR ${OptiX_SDK_DIR}/cuda)

include(FindCUDA)

#####################################################################################
# SUtil compilation
#

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
#set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SAMPLES_DIR "${PROJECT_SOURCE_DIR}/source")
set(DATA_DIR "${PROJECT_SOURCE_DIR}/data")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if (WIN32)
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else (WIN32)
  if ( USING_GNU_C AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif (WIN32)

# NVRTC include paths relative to the sample path
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"/sutil\", \\
  \"/cuda\", ")

# NVRTC absolute include paths to the headers used to build the samples
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${OptiX_INCLUDE}/optixu\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

include(CompilerInfo)
include(Macros)
include(ConfigCompilerFlags)

#
# Q: Why do i have to generate it twice in order to get the optixu path included into my sutil project? 
#
# NVRTC absolute include paths to the headers used to build the samples
set(SAPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${OptiX_INCLUDE}/optixu\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set(CUDA_NVRTC_OPTIONS)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  0,")

# Path to sutil.h that all the samples need
include_directories( #${CMAKE_CURRENT_SOURCE_DIR}/sutil
                     ${CMAKE_CURRENT_SOURCE_DIR}
                     ${OptiX_INCLUDE}/optixu
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS} )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

#####################################################################################

include(FindOptiX)

# 
# Sources
# 

#set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include/${target}")
set(source_path  "${CMAKE_CURRENT_SOURCE_DIR}/source")

#set(headers
#)

set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}")
set(OptiX_SDK_DIR "${OptiX_INSTALL_DIR}/SDK")
set(SAMPLES_CUDA_DIR ${OptiX_SDK_DIR}/cuda)

#include(${OptiX_SDK_DIR}/sampleConfig.h.in)

#configure_file(${OptiX_SDK_DIR}/sampleConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/cmake/sampleConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h @ONLY)

set(sources
	  ${SAMPLES_CUDA_DIR}/commonStructs.h
	  ${SAMPLES_CUDA_DIR}/helpers.h
	  ${SAMPLES_CUDA_DIR}/intersection_refinement.h
	  ${SAMPLES_CUDA_DIR}/phong.h
	  ${SAMPLES_CUDA_DIR}/phong.cu
	  ${SAMPLES_CUDA_DIR}/random.h
	  ${SAMPLES_CUDA_DIR}/triangle_mesh.cu
	  rply-1.01/rply.c
	  rply-1.01/rply.h
	  Arcball.cpp
	  Arcball.h
	  HDRLoader.cpp
	  HDRLoader.h
	  Mesh.cpp
	  Mesh.h
	  OptiXMesh.cpp
	  OptiXMesh.h
	  PPMLoader.cpp
	  PPMLoader.h
	  ${CMAKE_CURRENT_BINARY_DIR}/sampleConfig.h
	  sutil.cpp
	  sutil.h
	  sutilapi.h
	  tinyobjloader/tiny_obj_loader.cc
	  tinyobjloader/tiny_obj_loader.h
	  ${CMAKE_CURRENT_SOURCE_DIR}/LodePNG/lodepng.h
	  ${CMAKE_CURRENT_SOURCE_DIR}/LodePNG/lodepng.cpp
	  PNGLoader.h
	  PNGLoader.cpp
)

message("${sources}")

include(FindSUtilGLUT)

if(OPENGL_FOUND AND NOT APPLE)
  list(APPEND sources "glew.c" "GL/glew.h")
  if( WIN32 )
    list(APPEND sources "GL/wglew.h")
  else()
    list(APPEND sources "GL/glxew.h")
  endif()

  if( WIN32 )
    add_definitions(-DGLEW_BUILD)
  endif()

  if(USING_GNU_CXX OR USING_CLANG_CXX)
    # glew.c #includes glew.h which contains "lax" function prototypes
    set_source_files_properties("glew.c" PROPERTIES COMPILE_FLAGS  "-Wno-strict-prototypes -Wno-missing-prototypes")

  endif()
  
endif()

if(GLUT_FOUND AND OPENGL_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
  add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

endif()

# Group source files
#set(header_group "Header Files (API)")
#set(source_group "Source Files")
#source_group_by_path(${include_path} "\\\\.h$|\\\\.hpp$" 
#    ${header_group} ${headers})
#source_group_by_path(${source_path}  "\\\\.cpp$|\\\\.c$|\\\\.h$|\\\\.hpp$" 
#    ${source_group} ${sources})


# 
# Create library
# 

# Build library
add_library(${target}
    ${sources}
    #${headers}
)

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")

if(NOT CUDA_NVRTC_ENABLED)
  CUDA_COMPILE_PTX(ptx_files sources)
endif()

# Make the library.
#SET(SUTIL_TARGET "SUTIL_SDK")
#IF(CUDA_NVRTC_ENABLED)
#  ADD_LIBRARY(${SUTIL_TARGET} ${SOURCES})
#ELSE()
#  ADD_LIBRARY(${SUTIL_TARGET} ${SOURCES} ${PTX_FILES})
#ENDIF()


# Create namespaced alias
add_library(${META_PROJECT_NAME}::${target} ALIAS ${target})

# Export library for downstream projects
export(TARGETS ${target} NAMESPACE ${META_PROJECT_NAME}:: FILE ${PROJECT_BINARY_DIR}/cmake/${target}/${target}-export.cmake)

# Create feature detection header
# Compilers: https://cmake.org/cmake/help/v3.1/variable/CMAKE_LANG_COMPILER_ID.html#variable:CMAKE_%3CLANG%3E_COMPILER_ID
# Feature: https://cmake.org/cmake/help/v3.1/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html

# Check for availability of module; use pre-generated version if not found
if (WriterCompilerDetectionHeaderFound)
    write_compiler_detection_header(
        FILE ${feature_file}
        PREFIX ${target_id}
        COMPILERS AppleClang Clang GNU MSVC
        FEATURES cxx_alignas cxx_alignof cxx_constexpr cxx_final cxx_noexcept cxx_nullptr cxx_sizeof_member cxx_thread_local
        VERSION 3.2
    )
else()
    file(
        COPY ${PROJECT_SOURCE_DIR}/source/codegeneration/${target}_features.h
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include/${target}
        USE_SOURCE_PERMISSIONS
    )
endif()

# Create API export header
generate_export_header(${target}
    EXPORT_FILE_NAME  ${export_file}
    EXPORT_MACRO_NAME ${export_macro}
)

generate_template_export_header(${target}
    ${target_id}
    ${template_export_file}
)


# 
# Project options
# 

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
    VERSION ${META_VERSION}
    SOVERSION ${META_VERSION_MAJOR}
)


# 
# Include directories
# 

target_include_directories(${target}
    PRIVATE
    ${PROJECT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/LodePNG
    ${OptiX_INSTALL_DIR}/include
    ${OptiX_INSTALL_DIR}/include/optixu
    ${OptiX_SDK_DIR}/support/freeglut/include
    ${SAMPLES_CUDA_DIR}

    PUBLIC
    ${DEFAULT_INCLUDE_DIRECTORIES}

    INTERFACE
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    #$<INSTALL_INTERFACE:include>
)


# 
# Libraries
# 

target_link_libraries(${target}
    PRIVATE
    #${OptiX_SDK_DIR}

    PUBLIC
    ${DEFAULT_LIBRARIES}

    INTERFACE
)


# 
# Compile definitions
# 

target_compile_definitions(${target}
    PRIVATE

    PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${target_id}_STATIC_DEFINE>
    ${DEFAULT_COMPILE_DEFINITIONS}

    INTERFACE
)


# 
# Compile options
# 

target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
)

# 
# Linker options
# 

target_link_libraries(${target}
    PRIVATE
    optix
  	optixu
  	${GLUT_LIBRARIES}
  	${OPENGL_LIBRARIES}

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}

    INTERFACE
)

#include(FindCUDA)

if(NOT CUDA_NVRTC_ENABLED)
  message("CUDA_NVRTC_ENABLED ${CUDA_NVRTC_ENABLED}")
  # Be cautious to pass the second parameter without '${}', else it will lead to an error with the cmake list sub command.
  CUDA_COMPILE_PTX(ptx_files sources)
endif()

set(CUDA_nvrtc_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64/nvrtc.lib")

#set(sutil_target "sutil_sdk")
if(CUDA_NVRTC_ENABLED)
  message("BLA!!!")
  target_link_libraries(${target} PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
  target_link_libraries(${target} PRIVATE winmm.lib)
endif()

#
# Target Health
#

perform_health_checks(
    ${target}
    ${sources}
    #${headers}
)


# 
# Deployment
# 

# Library
install(TARGETS ${target}
    EXPORT  "${target}-export"            COMPONENT dev
    RUNTIME DESTINATION ${INSTALL_BIN}    COMPONENT runtime
    LIBRARY DESTINATION ${INSTALL_SHARED} COMPONENT runtime
    ARCHIVE DESTINATION ${INSTALL_LIB}    COMPONENT dev
)

# Header files
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${target} DESTINATION ${INSTALL_INCLUDE}
    COMPONENT dev
)

# Generated header files
install(DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/include/${target} DESTINATION ${INSTALL_INCLUDE}
    COMPONENT dev
)

# CMake config
install(EXPORT ${target}-export
    NAMESPACE   ${META_PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKE}/${target}
    COMPONENT   dev
)

##################################################################
# This copies out dlls into the build directories, so that users no longer need to copy
# them over in order to run the samples.  This depends on the optixHello sample being compiled.
# If you remove this sample from the list of compiled samples, then you should change
# "optixHello" found below to the name of one of your other samples.
#if(WIN32)
#  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
#    set(bit_dest "64")
#  else()
#    set(bit_dest "")
#  endif()
#  foreach(config ${CMAKE_CONFIGURATION_TYPES})
#    cmake_policy(SET CMP0026 OLD)  # disable warning about LOCATION property
#    get_target_property(loc optixHello ${config}_LOCATION)
#    if(loc)
#      # A little helper function
#      function(copy_dll lib)
#        get_filename_component(path ${loc} PATH)
#        get_filename_component(name ${lib} NAME)
#        #message("${CMAKE_COMMAND} -E copy_if_different ${lib} ${path}/${name}")
#        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} ${path}/${name})
#      endfunction()
#      
#      # Copy the binary directory into the build directory
#      file(GLOB dlls "${OptiX_INSTALL_DIR}/bin${bit_dest}/*.dll")
#      foreach(file ${dlls})
#        copy_dll("${file}")
#      endforeach()
#
#      # Copy the shipped MDL DLLs into the build directory if we didn't build our own
#      if(NOT USE_OWN_MDL_WRAPPER)
#        file(GLOB dlls "${OptiX_INSTALL_DIR}/SDK/support/mdl_wrapper/lib/*.dll")
#        foreach(file ${dlls})
#          copy_dll("${file}")
#        endforeach()
#      endif()
#    else()
#      message(WARNING "Unable to find location to copy DLLs into the build")
#    endif()
#  endforeach()
#endif(WIN32)