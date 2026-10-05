#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ICountPass.h"
#include "FloatToIntPass.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  my::registerICountPass();
  my::registerFloatToIntPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "My MLIR study tool\n", registry));
}
