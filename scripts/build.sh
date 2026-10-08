#!/bin/bash

set -e

# Load and validate the environment.
source "$(dirname "${BASH_SOURCE[0]}")/env_check.sh"

# Configure the build directory.
mkdir -p build
build_dir="$(cd build && pwd -P)"

CC="$LLVM_BUILD_DIR/bin/clang"
CXX="$LLVM_BUILD_DIR/bin/clang++"

# Configure the project.
cmake -S . -B "$build_dir" -G Ninja \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Compile the project.
cmake --build "$build_dir"
