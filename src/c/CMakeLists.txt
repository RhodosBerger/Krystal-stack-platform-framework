cmake_minimum_required(VERSION 3.16)
project(gamesa_core VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_OPENGL "Enable OpenGL support" ON)
option(ENABLE_VULKAN "Enable Vulkan support" ON)

# Find dependencies
find_package(Threads REQUIRED)
find_package(PkgConfig)

if(ENABLE_OPENGL)
    find_package(OpenGL)
endif()

if(ENABLE_VULKAN)
    find_package(Vulkan)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Kernel library
add_library(gamesa_kernel
    gamesa_kernel.c
)
target_link_libraries(gamesa_kernel PRIVATE Threads::Threads)

# Thread boost layer
add_library(thread_boost_layer
    thread_boost_layer.c
)
target_link_libraries(thread_boost_layer PRIVATE gamesa_kernel Threads::Threads)

# RPG craft system
add_library(rpg_craft_system
    rpg_craft_system.c
)
target_link_libraries(rpg_craft_system PRIVATE gamesa_kernel)

# Main core library combining all
add_library(gamesa_core INTERFACE)
target_link_libraries(gamesa_core INTERFACE
    gamesa_kernel
    thread_boost_layer
    rpg_craft_system
)

# Demo executables
add_executable(memory_tier_demo demos/memory_tier_demo.c)
target_link_libraries(memory_tier_demo PRIVATE gamesa_core)

add_executable(system_generator_demo demos/system_generator_demo.c)
target_link_libraries(system_generator_demo PRIVATE gamesa_core)

# Install
install(TARGETS gamesa_kernel thread_boost_layer rpg_craft_system
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    gamesa_kernel.h
    thread_boost_layer.h
    rpg_craft_system.h
    DESTINATION include/gamesa
)
