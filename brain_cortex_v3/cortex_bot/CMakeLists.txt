cmake_minimum_required(VERSION 3.18)
project(cortex_bot VERSION 3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags optimised for Tiger Lake ───────────────────
# -march=tigerlake enables VNNI, AVX512_BF16, AVX512VBMI2
# Fall back to native if tigerlake is not recognised by older GCC
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=tigerlake" COMPILER_HAS_TIGERLAKE)
if(COMPILER_HAS_TIGERLAKE)
  set(HW_ARCH_FLAGS "-march=tigerlake -mtune=tigerlake")
else()
  set(HW_ARCH_FLAGS "-march=native")
endif()

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 ${HW_ARCH_FLAGS} -ffast-math -funroll-loops -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g3 -fsanitize=address -fno-omit-frame-pointer")

# ── Optional: OpenVINO ────────────────────────────────────────
option(WITH_OPENVINO "Link against OpenVINO runtime" ON)

if(WITH_OPENVINO)
  find_package(OpenVINO QUIET)
  if(OpenVINO_FOUND)
    message(STATUS "OpenVINO ${OpenVINO_VERSION} found")
    add_compile_definitions(CORTEX_OV_AVAILABLE=1)
  else()
    message(WARNING "OpenVINO not found — building in stub mode")
    add_compile_definitions(CORTEX_NO_OPENVINO=1)
  endif()
else()
  add_compile_definitions(CORTEX_NO_OPENVINO=1)
endif()

# ── pthread (for future virtual-NPU thread pinning) ──────────
find_package(Threads REQUIRED)

# ── The executable ────────────────────────────────────────────
add_executable(cortex_bot
    src/main.cpp
)

target_include_directories(cortex_bot PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cortex_bot PRIVATE
    Threads::Threads
)

if(OpenVINO_FOUND)
  target_link_libraries(cortex_bot PRIVATE openvino::runtime)
endif()

# ── Install ───────────────────────────────────────────────────
install(TARGETS cortex_bot DESTINATION bin)

# ── Summary ───────────────────────────────────────────────────
message(STATUS "──────────────────────────────────────────")
message(STATUS "  Brain Cortex V3 — CortexBot Build")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  HW arch     : ${HW_ARCH_FLAGS}")
message(STATUS "  OpenVINO    : ${OpenVINO_FOUND}")
message(STATUS "  C++ std     : C++${CMAKE_CXX_STANDARD}")
message(STATUS "──────────────────────────────────────────")
