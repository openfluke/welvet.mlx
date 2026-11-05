cmake_minimum_required(VERSION 3.20)
project(welvet_mlx VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT APPLE)
  message(FATAL_ERROR "welvet.mlx requires an Apple host toolchain (macOS/iOS)")
endif()

if(IOS)
  set(WELVET_MLX_PLATFORM_IOS ON)
else()
  set(WELVET_MLX_PLATFORM_MACOS ON)
endif()

option(WELVET_MLX_BUILD_TOOLS "Build smoke executable" ON)
option(WELVET_MLX_BUILD_SHARED "Build shared libwelvet_mlx (macOS)" ON)

# iOS: static only, no host smoke binary.
if(IOS)
  set(WELVET_MLX_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(WELVET_MLX_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
endif()

# Prefer Homebrew MLX on macOS; on iOS require MLX_ROOT with libmlx.a (iphoneos).
set(MLX_ROOT "$ENV{MLX_ROOT}" CACHE PATH "Apple MLX install prefix")
if(NOT MLX_ROOT AND DEFINED ENV{MLX_DIR})
  set(MLX_ROOT "$ENV{MLX_DIR}" CACHE PATH "Apple MLX install prefix" FORCE)
endif()
if(NOT MLX_ROOT AND NOT IOS)
  foreach(cand /opt/homebrew /usr/local)
    if(EXISTS "${cand}/include/mlx/mlx.h" AND EXISTS "${cand}/lib/libmlx.dylib")
      set(MLX_ROOT "${cand}" CACHE PATH "Apple MLX install prefix" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT MLX_ROOT OR NOT EXISTS "${MLX_ROOT}/include/mlx/mlx.h")
  message(FATAL_ERROR
    "welvet.mlx: MLX not found.\n"
    "  macOS: brew install mlx  (or set MLX_ROOT)\n"
    "  iOS:   build MLX for iphoneos and set MLX_ROOT to that prefix (libmlx.a + headers).\n"
    "         See scripts/build_ios.sh / scripts/build_mlx_ios.sh")
endif()
message(STATUS "welvet.mlx: MLX at ${MLX_ROOT} (ios=${IOS})")

if(IOS)
  # Device link: prefer static archive (find_library often skips .a on iOS).
  if(EXISTS "${MLX_ROOT}/lib/libmlx.a")
    set(MLX_LIB "${MLX_ROOT}/lib/libmlx.a")
  else()
    find_library(MLX_LIB
      NAMES mlx
      HINTS "${MLX_ROOT}/lib"
      NO_DEFAULT_PATH
      REQUIRED
    )
  endif()
  message(STATUS "welvet.mlx: linking ${MLX_LIB}")
else()
  find_library(MLX_LIB mlx HINTS "${MLX_ROOT}/lib" REQUIRED)
endif()

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate)

set(WMLX_SOURCES
  src/detect.cpp
  src/dtype.cpp
  src/session.cpp
  src/prepared.cpp
  src/layers.cpp
  src/sequential.cpp
  src/c_abi.cpp
)

add_library(welvet_mlx STATIC ${WMLX_SOURCES})
add_library(welvet_mlx::welvet_mlx ALIAS welvet_mlx)

set(_wmlx_targets welvet_mlx)

if(WELVET_MLX_BUILD_SHARED)
  add_library(welvet_mlx_shared SHARED ${WMLX_SOURCES})
  set_target_properties(welvet_mlx_shared PROPERTIES
    OUTPUT_NAME welvet_mlx
    MACOSX_RPATH ON
    BUILD_WITH_INSTALL_RPATH OFF
    INSTALL_NAME_DIR "@rpath"
  )
  target_compile_definitions(welvet_mlx_shared PRIVATE WELVET_MLX_BUILD_SHARED)
  list(APPEND _wmlx_targets welvet_mlx_shared)
endif()

foreach(tgt ${_wmlx_targets})
  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      "${MLX_ROOT}/include"
  )
  target_compile_definitions(${tgt} PUBLIC
    WELVET_MLX=1
    $<$<BOOL:${WELVET_MLX_PLATFORM_MACOS}>:WELVET_OS_MACOS=1>
    $<$<BOOL:${WELVET_MLX_PLATFORM_IOS}>:WELVET_OS_IOS=1>
  )
  target_link_libraries(${tgt} PUBLIC
    ${MLX_LIB}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
  )
  if(ACCELERATE_FRAMEWORK)
    target_link_libraries(${tgt} PUBLIC ${ACCELERATE_FRAMEWORK})
  endif()
  if(NOT IOS)
    # Keep MLX dylib findable next to the shared lib / via brew (macOS only).
    target_link_options(${tgt} PUBLIC "LINKER:-rpath,${MLX_ROOT}/lib")
  endif()
endforeach()

if(WELVET_MLX_BUILD_TOOLS)
  add_executable(welvet_mlx_smoke tools/smoke/main.cpp)
  target_link_libraries(welvet_mlx_smoke PRIVATE welvet_mlx)
endif()

install(TARGETS ${_wmlx_targets}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/welvet_mlx DESTINATION include)
install(FILES include/welvet_mlx_c.h DESTINATION include)

message(STATUS "welvet.mlx platform: ${CMAKE_SYSTEM_NAME} shared=${WELVET_MLX_BUILD_SHARED}")
