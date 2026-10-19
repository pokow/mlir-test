#include "my/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
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

    // Materializes values
    auto bitcast = [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value
    {
      if (inputs.size() != 1)
      {
        return nullptr;
      }
      return builder.create<stablehlo::BitcastConvertOp>(loc, resultType, inputs[0]);
    };

    addSourceMaterialization(bitcast); // int -> float
    addTargetMaterialization(bitcast); // float -> int
  }
};

// Check if an operation is constant
bool isConstantLikeOp(Operation *op) 
{
  return op->hasTrait<OpTrait::ConstantLike>();
}

class FloatToIntPass : public impl::FloatToIntBase<FloatToIntPass>
{
public:

  void runOnOperation() override
  {
    // Get current func::FuncOp
    auto funcOp = getOperation();

    // Get FuncOp context
    auto *ctx = funcOp->getContext();

    FloatToIntTypeConverter typeConverter;
    IRRewriter rewriter(ctx);

    // Update function signature
    for (BlockArgument arg : funcOp.getArguments())
    {
      arg.setType(typeConverter.convertType(arg.getType()));
    }

    // Update function input and output type list
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;

    for (Type t : funcOp.getArgumentTypes())
    {
      newInputs.push_back(typeConverter.convertType(t));
    }

    for (Type t : funcOp.getResultTypes())
    {
      newResults.push_back(typeConverter.convertType(t));
    }

    // Override original type of the function
    funcOp.setFunctionType(rewriter.getFunctionType(newInputs, newResults));

    // Collect the operations (AST walk)
    SmallVector<Operation *> constantOpsToReplace;
    SmallVector<Operation *> arithOpsToReplace;

    funcOp.walk([&](Operation *op)
    {
      if (auto cst = llvm::dyn_cast<arith::ConstantOp>(op))
      {
        if (typeConverter.convertType(cst.getType()) != cst.getType())
        {
          constantOpsToReplace.push_back(op);
        }
      } 
      else if (llvm::isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op))
      {
        arithOpsToReplace.push_back(op);
      }
    });

    // Replace Constant Operations
    for (Operation *op : constantOpsToReplace)
    {
      rewriter.setInsertionPoint(op);

      Type newType = typeConverter.convertType(op->getResult(0).getType());
      Attribute valueAttr = op->getAttr("value");
      TypedAttr newAttr;

      if (auto floatAttr = llvm::dyn_cast<FloatAttr>(valueAttr))
      {
        // Scalar: reinterpret the float bits as an integer
        APInt bits = floatAttr.getValue().bitcastToAPInt();
        newAttr = IntegerAttr::get(newType, bits);
      }
      else if (auto denseAttr = llvm::dyn_cast<DenseFPElementsAttr>(valueAttr))
      {
        // Tensor: reinterpret the bits of each float element
        auto intType = llvm::cast<ShapedType>(newType).getElementType();

        SmallVector<APInt> values;
        for (const APFloat &floatValue : denseAttr.getValues<APFloat>())
        {
          values.push_back(floatValue.bitcastToAPInt());
        }

        newAttr = DenseIntElementsAttr::get(llvm::cast<ShapedType>(newType), values);
      }

      auto newConstant = rewriter.create<arith::ConstantOp>(op->getLoc(), newAttr);

      op->replaceAllUsesWith(newConstant);
      rewriter.eraseOp(op);
    }

    // Replace Arithmetic Operations
    for (Operation *op : arithOpsToReplace) 
    {
      rewriter.setInsertionPoint(op);
      Location loc = op->getLoc();

      Value lhs = op->getOperand(0);
      Value rhs = op->getOperand(1);

      Operation *newOp = nullptr;

      if (llvm::isa<arith::AddFOp>(op))
      {
        newOp = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
      }
      else if (llvm::isa<arith::SubFOp>(op))
      {
        newOp = rewriter.create<arith::SubIOp>(loc, lhs, rhs);
      }
      else if (llvm::isa<arith::MulFOp>(op))
      {
        newOp = rewriter.create<arith::MulIOp>(loc, lhs, rhs);
      }

      for (size_t idx = 0; idx < op->getNumResults(); idx++)
      {
        op->getResult(idx).replaceAllUsesWith(newOp->getResult(idx));
      }

      rewriter.eraseOp(op);
    }
  }
};

} // namespace
} // namespace my