cmake_minimum_required (VERSION 3.20)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.24.3.tar.gz"
    SHA1 "10738b59e539818a01090e64c2d09896247530c7"
)

project (vulkan_ethminer)
set (CMAKE_CXX_STANDARD 17)

#add package from hunter
hunter_add_package(asio)
hunter_add_package(spdlog)
hunter_add_package(RapidJSON)
hunter_add_package(CLI11)
find_package(asio CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

#Please set VULKAN_SDK
find_package(Vulkan REQUIRED COMPONENTS glslc)
message("vulkan lib: ${Vulkan_LIBRARY}")
message("vulkan inc: ${Vulkan_INCLUDE_DIR}")
include_directories(${Vulkan_INCLUDE_DIR})

#add ethash/etcash lib
include(ExternalProject)
ExternalProject_Add(ethash_git
        GIT_REPOSITORY https://github.com/mshzhb/cpp-etchash.git
        GIT_TAG main
        UPDATE_COMMAND ""
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern_install/ethash
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        )

SET(ETHASH_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern_install/ethash/include)
SET(ETHASH_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern_install/ethash/lib)
include_directories(${ETHASH_INCLUDE_DIR})
add_library(ethash STATIC IMPORTED)
add_library(ethash-global-context STATIC IMPORTED)
add_library(keccak STATIC IMPORTED)
IF (WIN32)
    set_target_properties(ethash PROPERTIES IMPORTED_LOCATION ${ETHASH_LIB_DIR}/ethash.lib)
    set_target_properties(keccak PROPERTIES IMPORTED_LOCATION ${ETHASH_LIB_DIR}/keccak.lib)
ELSE()
    set_target_properties(ethash PROPERTIES IMPORTED_LOCATION ${ETHASH_LIB_DIR}/libethash.a)
    set_target_properties(keccak PROPERTIES IMPORTED_LOCATION ${ETHASH_LIB_DIR}/libkeccak.a)
ENDIF()
#==============================================================================
# COMPILE SHADERS
set(GLSL_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/glsl)
set(HLSL_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/hlsl)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
find_program(dxc_executable NAMES dxc HINTS Vulkan::dxc)
message("project src dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message("glslc path: ${glslc_executable}")
message("dxc path: ${dxc_executable}")

#helper function
function(calc_power_of_two_exponent num exponent)
    set(counter 0)
    # Shift until our number equals 1.
    while(num GREATER 1)
        # Right shift by 1
        math(EXPR num "${num} >> 1")
        # Count the number of times we shift.
        math(EXPR counter "${counter} + 1")
    endwhile()
    # Return the number of times shifted, which is the exponent.
    set(exponent ${counter} PARENT_SCOPE)
endfunction()

#macro
set(DEBUG 0)
set(MAX_BUFFER_SIZE 0x80000000) #2GB
set(SINGLE_BUFFER_MODE 0) #only can enable this when MAX_BUFFER_SIZE is 4GB

#log2 MAX_BUFFER_SIZE - 6 // 2^6 == 64
calc_power_of_two_exponent(${MAX_BUFFER_SIZE} exponent)
math(EXPR exponent "${exponent} - 6")
set(DATASET_SELECT_BIT ${exponent})
add_compile_definitions(MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE})
message("MAX_BUFFER_SIZE: ${MAX_BUFFER_SIZE} bytes")
message("DATASET_SELECT_BIT: ${DATASET_SELECT_BIT}")

file(GLOB COMP_SHADERS  ${HLSL_SHADER_SOURCE_DIR}/ETHashGenerateDataset.hlsl
                        ${HLSL_SHADER_SOURCE_DIR}/ETHashMine_sharedMemory.hlsl
                        ${HLSL_SHADER_SOURCE_DIR}/ETHashMine_waveShuffle.hlsl)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

foreach(source IN LISTS COMP_SHADERS)
    get_filename_component(FILENAME ${source} NAME)
    if(DEBUG)
        add_custom_command(
                COMMAND
                ${dxc_executable}
                -T cs_6_5
                -E main
                -O3
                -D DEBUG
                -D DATASET_SELECT_BIT=${DATASET_SELECT_BIT}
                -D SINGLE_BUFFER_MODE=${SINGLE_BUFFER_MODE}
                -fspv-target-env=vulkan1.1
                ${source}
                -Fo ${SHADER_BINARY_DIR}/${FILENAME}.cso
                -spirv
                OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.cso
                DEPENDS ${source} ${SHADER_BINARY_DIR}
                COMMENT "Compiling ${FILENAME}"
        )
    else()
        add_custom_command(
                COMMAND
                ${dxc_executable}
                -T cs_6_5
                -E main
                -O3
                -D DATASET_SELECT_BIT=${DATASET_SELECT_BIT}
                -D SINGLE_BUFFER_MODE=${SINGLE_BUFFER_MODE}
                -fspv-target-env=vulkan1.1
                ${source}
                -Fo ${SHADER_BINARY_DIR}/${FILENAME}.cso
                -spirv
                OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.cso
                DEPENDS ${source} ${SHADER_BINARY_DIR}
                COMMENT "Compiling ${FILENAME}"
        )
    endif()
    list(APPEND HLSL_SHADERS ${SHADER_BINARY_DIR}/${FILENAME}.cso)
endforeach()
add_custom_target(shaders ALL DEPENDS ${HLSL_SHADERS})

if(DEBUG)
    add_compile_definitions(DEBUG)
    add_compile_definitions(TEST_MODE)
endif()

add_executable( vulkan_ethminer
                src/main.cpp
                src/lib-model/node.cpp
                src/lib-hash/sha3.cpp
                src/miner/vulkanminer.cpp 
                src/stratum/stratum.cpp src/monitor/monitor.cpp src/monitor/monitor.h)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(vulkan_ethminer PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
target_link_libraries(vulkan_ethminer ${Vulkan_LIBRARY} ethash keccak asio::asio_static spdlog::spdlog RapidJSON::rapidjson CLI11::CLI11)
add_dependencies(vulkan_ethminer shaders)
add_dependencies(vulkan_ethminer ethash_git)

