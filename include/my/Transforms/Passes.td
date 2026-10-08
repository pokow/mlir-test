#ifndef MY_TRANSFORMS_PASSES
#define MY_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FloatToInt : Pass<"float-to-int", "mlir::func::FuncOp">
{
  let summary = "Converts float types to integer types";
  let description = [{ Converts float tensors (and related arith ops) to integer tensors. }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def ICount : Pass<"icount", "mlir::ModuleOp">
{
  let summary = "Counts the number of operations in a module";
}

#endif // MY_TRANSFORMS_PASSES