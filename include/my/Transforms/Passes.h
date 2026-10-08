#ifndef MY_TRANSFORMS_PASSES_H
#define MY_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace my
{

#define GEN_PASS_DECL
#include "my/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "my/Transforms/Passes.h.inc"

}

#endif