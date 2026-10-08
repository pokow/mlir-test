add_mlir_library (
  MyTransforms
  FloatToInt.cpp
  ICount.cpp

  DEPENDS
  MyTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)