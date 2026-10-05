#include "ICountPass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace my
{

void ICountPass::runOnOperation()
{
  int64_t count = 0;

  auto ops = getOperation();

  ops.walk([&](mlir::Operation* op)
  {
    count += 1;
  });

  llvm::outs() << "Instruction count: " << count << "\n";
}


void registerICountPass()
{
  PassRegistration<ICountPass>(); 
} 


std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createICountPass()
{
  return std::make_unique<ICountPass>();
}

}