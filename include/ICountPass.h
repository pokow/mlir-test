#ifndef ICOUNT_PASS_H
#define ICOUNT_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace my
{

class ICountPass
  : public mlir::PassWrapper<ICountPass, mlir::OperationPass<mlir::ModuleOp>>
{

private:
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "icount"; }

  llvm::StringRef getDescription() const final
  {
    return "Counts the number of instrucrions of a file.mlir";
  }

};

void registerICountPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createICountPass();

}

#endif