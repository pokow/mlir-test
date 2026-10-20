#include "my/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <vector>

using namespace mlir;

namespace my
{

#define GEN_PASS_DEF_FLOATTOINT
#include "my/Transforms/Passes.h.inc"

namespace
{

// TypeConverter: fN -> iN, tensor<...xfN> -> tensor<...xiN>
class FloatToIntTypeConverter : public TypeConverter
{
public:
  FloatToIntTypeConverter()
  {
    // Keeps types that do not need to be converted.
    addConversion([](Type t) -> Type
    {
      return t;
    });

    // Converts any float type to an integer type of the same width.
    addConversion([](FloatType f)-> Type
    {
      return IntegerType::get(f.getContext(), f.getWidth());
    });

    // Converts the element type of ranked tensors.
    addConversion([this](RankedTensorType rt) -> Type
    {
      Type elem = convertType(rt.getElementType());
      if (!elem)
      {
        return nullptr;
      }
      return RankedTensorType::get(rt.getShape(), elem, rt.getEncoding());
    });
  }
};

// Converts signature and entry block argument types (fN -> iN).
// Leaves the IR invalid until convertBody fixes the uses.
void convertFunction(func::FuncOp &funcOp,
                      FloatToIntTypeConverter &typeConverter,
                      IRRewriter &rewriter)
{
  FunctionType oldFuncType = funcOp.getFunctionType();

  // Compute new type lists first
  SmallVector<Type> newInputs;
  SmallVector<Type> newResults;

  for (Type t : oldFuncType.getInputs())
  {
    Type newT = typeConverter.convertType(t);
    newInputs.push_back(newT);
  }

  for (Type t : oldFuncType.getResults())
  {
    Type newT = typeConverter.convertType(t);
    newResults.push_back(newT);
  }

  for (BlockArgument arg : funcOp.getArguments())
  {
    arg.setType(typeConverter.convertType(arg.getType()));
  }

  funcOp.setFunctionType(rewriter.getFunctionType(newInputs, newResults));
}

void convertCall(func::CallOp &callOp,
                 FloatToIntTypeConverter &typeConverter,
                 IRRewriter &rewriter)
{
  // Change the type of the arguments
  for (Value argument : callOp->getOperands())
  {
    argument.setType(typeConverter.convertType(argument.getType()));
  }
  // Change the type of the results
  for (Value result : callOp.getResults())
  {
    result.setType(typeConverter.convertType(result.getType()));
  }
}

class FloatToIntPass : public impl::FloatToIntBase<FloatToIntPass>
{
public:

  void runOnOperation() override
  {
    // Get current ModuleOp
    auto moduleOp = getOperation();

    // Get current ModuleOp context
    auto *ctx = moduleOp->getContext();

    // Instance FloatToIntTypeConverter
    FloatToIntTypeConverter typeConverter;

    // Instance IRRewriter
    IRRewriter rewriter(ctx);

    // Collect all FuncOps and CallOps (before mutating the IR)
    SmallVector<func::FuncOp> funcOps;
    SmallVector<func::CallOp> callOps;

    moduleOp.walk([&](Operation *op)
    {
      if (auto funcOp = dyn_cast<func::FuncOp>(op))
      {
        funcOps.push_back(funcOp);
      }
      else if (auto callOp = dyn_cast<func::CallOp>(op))
      {
        callOps.push_back(callOp);
      }
    });

    // Convert FuncOps first (signature, entry block, returns)
    for (func::FuncOp funcOp : funcOps)
    {
      convertFunction(funcOp, typeConverter, rewriter);
    }

    // Then convert CallOps (callees already have the new signature)
    for (func::CallOp callOp : callOps)
    {
      convertCall(callOp, typeConverter, rewriter);
    }

    // Check that the IR is still valid
    // if (failed(verify(moduleOp)))
    // {
    //   signalPassFailure();
    // }

    // Etapa 2 (depois): corpo
    // for (func::FuncOp funcOp : funcOps)
    // {
    //   convertBody(funcOp, typeConverter, rewriter);
    // }
  }
};

} // namespace
} // namespace my