cmake_minimum_required(VERSION 3.20)
project(my-opt)

set(MLIR_DIR /home/pedrogarcia/llvm-project/build/lib/cmake/mlir)

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(my-opt
  main/my-opt.cpp
  src/ICountPass.cpp
  src/FloatToIntPass.cpp
)

llvm_update_compile_flags(my-opt)

target_link_libraries(my-opt
  PRIVATE
  MLIROptLib
  ${dialect_libs}
  ${conversion_libs}
)

target_include_directories(my-opt
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

mlir_check_all_link_libraries(my-opt)
