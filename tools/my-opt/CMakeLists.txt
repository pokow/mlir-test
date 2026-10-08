get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(my-opt my-opt.cpp)
llvm_update_compile_flags(my-opt)

target_link_libraries(
  my-opt
  PRIVATE
  MLIROptLib
  MyTransforms
  ${dialect_libs}
  ${conversion_libs}
)

mlir_check_all_link_libraries(my-opt)