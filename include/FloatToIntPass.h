#ifndef FLOAT_TO_INT_PASS_H
#define FLOAT_TO_INT_PASS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace my 
{

class FloatToIntPass
  : public mlir::PassWrapper<FloatToIntPass, mlir::OperationPass<mlir::func::FuncOp>>
{
public:

  llvm::StringRef getArgument() const final { return "float-to-int"; }

  llvm::StringRef getDescription() const final
  {
    return "Converts float tensors (and related arith ops) to  int tensors, zeroing constant values.";
  }

private:

  void runOnOperation() override;
};

void registerFloatToIntPass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createFloatToIntPass();

}

#endif