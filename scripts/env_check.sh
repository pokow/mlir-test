#!/bin/bash

# Locate LLVM if LLVM_BUILD_DIR was not provided.
if [ -z "$LLVM_BUILD_DIR" ]; then
    for dir in \
        "$HOME/llvm-project/build" \
        "/usr/lib/llvm-21" \
        "/usr/local"
    do
        if [ -d "$dir/lib/cmake/mlir" ] &&
           [ -x "$dir/bin/clang" ] &&
           [ -x "$dir/bin/clang++" ]; then
            LLVM_BUILD_DIR="$dir"
            break
        fi
    done
fi

# Validate the LLVM/MLIR installation.
if [ -z "$LLVM_BUILD_DIR" ] ||
   [ ! -d "$LLVM_BUILD_DIR/lib/cmake/mlir" ] ||
   [ ! -d "$LLVM_BUILD_DIR/lib/cmake/llvm" ] ||
   [ ! -x "$LLVM_BUILD_DIR/bin/clang" ] ||
   [ ! -x "$LLVM_BUILD_DIR/bin/clang++" ]; then
    >&2 echo "Error: LLVM/MLIR installation not found or incomplete."
    >&2 echo "Set LLVM_BUILD_DIR to your LLVM build directory."
    >&2 echo "Example: export LLVM_BUILD_DIR=/path/to/llvm-project/build"
    exit 1
fi

# Export the detected path for scripts and subprocesses.
export LLVM_BUILD_DIR