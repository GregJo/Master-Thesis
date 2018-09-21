
# 
# CMake options
# 

# CMake version
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#
# Configure CMake environment
#

# Register general cmake commands
include(cmake/Custom.cmake)

# Set policies
set_policy(CMP0054 NEW) # ENABLE CMP0054: Only interpret if() arguments as variables or keywords when unquoted.
set_policy(CMP0042 NEW) # ENABLE CMP0042: MACOSX_RPATH is enabled by default.
set_policy(CMP0063 NEW) # ENABLE CMP0063: Honor visibility properties for all target types.

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GenerateExportHeader)

set(WriterCompilerDetectionHeaderFound NOTFOUND)
# This module is only available with CMake >=3.1, so check whether it could be found
# BUT in CMake 3.1 this module doesn't recognize AppleClang as compiler, so just use it as of CMake 3.2
if (${CMAKE_VERSION} VERSION_GREATER "3.2")
    include(WriteCompilerDetectionHeader OPTIONAL RESULT_VARIABLE WriterCompilerDetectionHeaderFound)
endif()

# Include custom cmake modules
include(cmake/GetGitRevisionDescription.cmake)
include(cmake/HealthCheck.cmake)
include(cmake/GenerateTemplateExportHeader.cmake)

# 
# Project description and (meta) information
# 

# Get git revision
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
string(SUBSTRING "${GIT_SHA1}" 0 12 GIT_REV)
if(NOT GIT_SHA1)
    set(GIT_REV "0")
endif()

# Meta information about the project
set(META_PROJECT_NAME        "thesis")
set(META_PROJECT_DESCRIPTION "My master thesis.")
set(META_AUTHOR_ORGANIZATION "")
set(META_AUTHOR_DOMAIN       "")
set(META_AUTHOR_MAINTAINER   "")
set(META_VERSION_MAJOR       "2")
set(META_VERSION_MINOR       "0")
set(META_VERSION_PATCH       "0")
set(META_VERSION_REVISION    "${GIT_REV}")
set(META_VERSION             "${META_VERSION_MAJOR}.${META_VERSION_MINOR}.${META_VERSION_PATCH}")
set(META_NAME_VERSION        "${META_PROJECT_NAME} v${META_VERSION} (${META_VERSION_REVISION})")
set(META_CMAKE_INIT_SHA      "${GIT_REV}")

string(MAKE_C_IDENTIFIER ${META_PROJECT_NAME} META_PROJECT_ID)
string(TOUPPER ${META_PROJECT_ID} META_PROJECT_ID)


# 
# Project configuration options
# 

# Project options
option(BUILD_SHARED_LIBS     "Build shared instead of static libraries."              ON)
option(OPTION_SELF_CONTAINED "Create a self-contained install with all dependencies." OFF)
option(OPTION_BUILD_TESTS    "Build tests."                                           ON)
option(OPTION_BUILD_DOCS     "Build documentation."                                   OFF)
option(OPTION_BUILD_EXAMPLES "Build examples."                                        OFF)


# 
# Declare project
# 

# Generate folders for IDE targets (e.g., VisualStudio solutions)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(IDE_FOLDER "")

# Declare project
project(${META_PROJECT_NAME} C CXX)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Create version file
file(WRITE "${PROJECT_BINARY_DIR}/VERSION" "${META_NAME_VERSION}")


# 
# Compiler settings and options
# 

include(cmake/CompileOptions.cmake)


#
# Project Health Check Setup
#

# Add cmake-init template check cmake targets
add_check_template_target(${META_CMAKE_INIT_SHA})

# Configure health check tools
enable_cppcheck(On)
enable_clang_tidy(On)


# 
# Deployment/installation setup
# 

# Get project name
set(project ${META_PROJECT_NAME})

# Check for system dir install
set(SYSTEM_DIR_INSTALL FALSE)
if("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr" OR "${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr/local")
    set(SYSTEM_DIR_INSTALL TRUE)
endif()

# Installation paths
if(UNIX AND SYSTEM_DIR_INSTALL)
    # Install into the system (/usr/bin or /usr/local/bin)
    set(INSTALL_ROOT      "share/${project}")       # /usr/[local]/share/<project>
    set(INSTALL_CMAKE     "share/${project}/cmake") # /usr/[local]/share/<project>/cmake
    set(INSTALL_EXAMPLES  "share/${project}")       # /usr/[local]/share/<project>
    set(INSTALL_DATA      "share/${project}")       # /usr/[local]/share/<project>
    set(INSTALL_BIN       "bin")                    # /usr/[local]/bin
    set(INSTALL_SHARED    "lib")                    # /usr/[local]/lib
    set(INSTALL_LIB       "lib")                    # /usr/[local]/lib
    set(INSTALL_INCLUDE   "include")                # /usr/[local]/include
    set(INSTALL_DOC       "share/doc/${project}")   # /usr/[local]/share/doc/<project>
    set(INSTALL_SHORTCUTS "share/applications")     # /usr/[local]/share/applications
    set(INSTALL_ICONS     "share/pixmaps")          # /usr/[local]/share/pixmaps
    set(INSTALL_INIT      "/etc/init")              # /etc/init (upstart init scripts)
else()
    # Install into local directory
    set(INSTALL_ROOT      ".")                      # ./
    set(INSTALL_CMAKE     "cmake")                  # ./cmake
    set(INSTALL_EXAMPLES  ".")                      # ./
    set(INSTALL_DATA      ".")                      # ./
    set(INSTALL_BIN       ".")                      # ./
    set(INSTALL_SHARED    "lib")                    # ./lib
    set(INSTALL_LIB       "lib")                    # ./lib
    set(INSTALL_INCLUDE   "include")                # ./include
    set(INSTALL_DOC       "doc")                    # ./doc
    set(INSTALL_SHORTCUTS "misc")                   # ./misc
    set(INSTALL_ICONS     "misc")                   # ./misc
    set(INSTALL_INIT      "misc")                   # ./misc
endif()

# Set runtime path
set(CMAKE_SKIP_BUILD_RPATH            FALSE) # Add absolute path to all dependencies for BUILD
set(CMAKE_BUILD_WITH_INSTALL_RPATH    FALSE) # Use CMAKE_INSTALL_RPATH for INSTALL
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE) # Do NOT add path to dependencies for INSTALL

if(NOT SYSTEM_DIR_INSTALL)
    # Find libraries relative to binary
    if(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path/../../../${INSTALL_LIB}")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/${INSTALL_LIB}")       
    endif()
endif()


# 
# Project modules
# 

add_subdirectory(source)
add_subdirectory(docs)


# 
# Deployment (global project files)
# 

# Install version file
install(FILES "${PROJECT_BINARY_DIR}/VERSION" DESTINATION ${INSTALL_ROOT} COMPONENT runtime)

# Install cmake find script for the project
install(FILES ${META_PROJECT_NAME}-config.cmake DESTINATION ${INSTALL_ROOT} COMPONENT dev)

# Install the project meta files
install(FILES AUTHORS   DESTINATION ${INSTALL_ROOT} COMPONENT runtime)
install(FILES LICENSE   DESTINATION ${INSTALL_ROOT} COMPONENT runtime)
install(FILES README.md DESTINATION ${INSTALL_ROOT} COMPONENT runtime)

# Install runtime data
install(DIRECTORY ${PROJECT_SOURCE_DIR}/data DESTINATION ${INSTALL_DATA} COMPONENT runtime)

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
#function(OPTIX_add_sample_executable target_name)
#
#  # These calls will group PTX and CUDA files into their own directories in the Visual
#  # Studio projects.
#  if (NOT CUDA_NVRTC_ENABLED)
#    source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
#  endif()
#  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
#
#  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
#  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
#  # code here, so that we can use our own name for the target.  target_name is used in the
#  # creation of the output file names, and we want this to be unique for each target in
#  # the SDK.
#  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})
#
#  if (CUDA_NVRTC_ENABLED)
#  
#    # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
#    set(cu_obj_source_files)
#    foreach(file ${source_files})
#      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
#      if(${_cuda_source_format} MATCHES "OBJ")
#        list(APPEND cu_obj_source_files ${file})
#      endif()
#    endforeach()
#
#    # Create the rules to build the OBJ from the CUDA files.
#    CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )
#  else()
#  
#    # Create the rules to build the PTX and OBJ from the CUDA files.
#    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
#  endif()
#  
#
#  # Here is where we create the rule to make the executable.  We define a target name and
#  # list all the source files used to create the target.  In addition we also pass along
#  # the cmake_options parsed out of the arguments.
#  add_executable(${target_name}
#    ${source_files}
#    ${generated_files}
#    ${cmake_options}
#    )
#
#  # Most of the samples link against the sutil library and the optix library.  Here is the
#  # rule that specifies this linkage.
#  target_link_libraries( ${target_name}
#    sutil_sdk
#    optix
#    ${optix_rpath}
#    )
#
#  if( UNIX AND NOT APPLE )
#    # Force using RPATH instead of RUNPATH on Debian
#    target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
#  endif()
#
#  if(USING_GNU_CXX)
#    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
#  endif()
#endfunction()
