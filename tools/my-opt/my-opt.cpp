#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "my/Transforms/Passes.h"

int main(int argc, char **argv)
{
  mlir::registerAllPasses();
  my::registerMyPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "My MLIR study tool\n", registry));
}
