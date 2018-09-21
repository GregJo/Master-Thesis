
# 
# External dependencies
# 


set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}")
set(OptiX_SDK_DIR "${OptiX_INSTALL_DIR}/SDK")

find_package(OptiX REQUIRED)
include_directories("${OptiX_INCLUDE}")

include(FindCUDA)
include(CompilerInfo)
include(FindSUtilGLUT)

# Enable automoc
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(AUTOMOC_MOC_OPTIONS PROPERTIES FOLDER CMakeAutomocTargets)
set_property(GLOBAL PROPERTY AUTOMOC_FOLDER CMakeAutomocTargets)


# 
# Executable name and options
# 

# Target name
set(target master-thesis)


# 
# Sources
# 

set(sources
    optixHello.cpp
    Scene.h
    Scenes.h
    Buffers.h
    optixPathTracer.h
    MitchellFilter.h
    MitchellFilterDevice.h
    Camera.h
    TrackballCamera.h
    HoelderAdaptive.h
    AdaptivePathTraceRenderContext.h
    IOptixRenderContext.h
    PathTraceRenderContext.h
    RayTraceRenderContext.h
    optixPathTracer.cu
    adaptiveOptixPathTracer.cu
    parallelogram.cu
)


# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")

if (NOT CUDA_NVRTC_ENABLED)
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
endif()
source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

if (CUDA_NVRTC_ENABLED)

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_sources)
  foreach(file ${sources})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if(${_cuda_source_format} MATCHES "OBJ")
      list(APPEND cu_obj_sources ${file})
    endif()
  endforeach()
  # Create the rules to build the OBJ from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_sources} ${cmake_options} OPTIONS ${options} )
else()
  # Create the rules to build the PTX and OBJ from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${sources} ${cmake_options} OPTIONS ${options} )
endif()

set(CUDA_cublas_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cublas.lib")
set(CUDA_cublas_device_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cublas_device.lib")
set(CUDA_cufft_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cufft.lib")
set(CUDA_curand_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\curand.lib")
set(CUDA_cusolver_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cuslover.lib")
set(CUDA_cusparse_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\cusparse.lib")
set(CUDA_nppc_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\nppc.lib")
set(CUDA_nppi_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\nppi.lib")
set(CUDA_npps_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\npps.lib")
set(CUDA_nvrtc_LIBRARY "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64\\nvrtc.lib")

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)


# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch sm_30)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()
  set(flag "--use_fast_math")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()

  if (CUDA_VERSION VERSION_LESS "3.0")
    set(flag "--keep")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    set(flag "-U__BLOCKS__")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()

  set(CUDA_NVRTC_FLAGS -arch compute_30 -use_fast_math -default-device -rdc true -D__x86_64 CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

mark_as_advanced(CUDA_NVRTC_FLAGS)

# 
# Create executable
#

# Build executable
add_executable(${target}
    #MACOSX_BUNDLE
    ${sources}
    ${generated_files}
    ${cmake_options}
)

# Create namespaced alias
add_executable(${META_PROJECT_NAME}::${target} ALIAS ${target})


# 
# Project options
# 

set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
)

# 
# Include directories
# 

target_include_directories(${target}
    PRIVATE
    ${DEFAULT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}
    #${PROJECT_BINARY_DIR}/source/include

    ${OptiX_INSTALL_DIR}/include
    ${OptiX_INSTALL_DIR}/include/optixu
    ${OptiX_SDK_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/source/sutil
    ${PROJECT_SOURCE_DIR}/source/sutil
    ${PROJECT_SOURCE_DIR}/source/support/freeglut/include
)


# 
# Libraries
# 

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LIBRARIES}
    optix
    optixu
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${META_PROJECT_NAME}::sutil_sdk
)

#set(sutil_target "sutil_sdk")
if(CUDA_NVRTC_ENABLED)
  target_link_libraries(${target} PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
  target_link_libraries(${target} PRIVATE winmm.lib)
endif()

# 
# Compile definitions
# 

target_compile_definitions(${target}
    PRIVATE
    ${DEFAULT_COMPILE_DEFINITIONS}

)


# 
# Compile options
# 

target_compile_options(${target}
    PRIVATE
    ${DEFAULT_COMPILE_OPTIONS}
)


# 
# Linker options
# 

target_link_libraries(${target}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
)


#
# Target Health
#

perform_health_checks(
    ${target}
    ${sources}
)


# 
# Deployment
# 

# Executable
install(TARGETS ${target}
    RUNTIME DESTINATION ${INSTALL_BIN} COMPONENT examples
    BUNDLE  DESTINATION ${INSTALL_BIN} COMPONENT examples
)

if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
    set(bit_dest "64")
  else()
    set(bit_dest "")
  endif()
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    cmake_policy(SET CMP0026 OLD)  # disable warning about LOCATION property
    get_target_property(loc master-thesis ${config}_LOCATION)
    if(loc)
      # A little helper function
      function(copy_dll lib)
        get_filename_component(path ${loc} PATH)
        get_filename_component(name ${lib} NAME)
        #message("${CMAKE_COMMAND} -E copy_if_different ${lib} ${path}/${name}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} ${path}/${name})
      endfunction()
      
      # Copy the binary directory into the build directory
      file(GLOB dlls "${OptiX_INSTALL_DIR}/bin${bit_dest}/*.dll")
      foreach(file ${dlls})
        copy_dll("${file}")
      endforeach()

      message("${OptiX_SDK_DIR}/support/freeglut/win${bit_dest}/Release")
      file(GLOB dlls "${OptiX_SDK_DIR}/support/freeglut/win${bit_dest}/Release/*.dll")
      foreach(file ${dlls})
        copy_dll("${file}")
      endforeach()

      # Copy the shipped MDL DLLs into the build directory if we didn't build our own
      if(NOT USE_OWN_MDL_WRAPPER)
        file(GLOB dlls "${OptiX_INSTALL_DIR}/SDK/support/mdl_wrapper/lib/*.dll")
        foreach(file ${dlls})
          copy_dll("${file}")
        endforeach()
      endif()
    else()
      message(WARNING "Unable to find location to copy DLLs into the build")
    endif()
  endforeach()
endif(WIN32)

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