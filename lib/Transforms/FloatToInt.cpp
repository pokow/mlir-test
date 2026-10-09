#include "my/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

using namespace mlir;

namespace my
{

#define GEN_PASS_DEF_FLOATTOINT
#include "my/Transforms/Passes.h.inc"

namespace
{

// Converts FloatType or RankedTensorType of floats to integer type
Type convertType(Type t)
{
  if (auto f = llvm::dyn_cast<FloatType>(t))
  {
    return IntegerType::get(t.getContext(), f.getWidth());
  }

  if (auto rt = llvm::dyn_cast<RankedTensorType>(t))
  {
    Type newElemType = convertType(rt.getElementType());
    return RankedTensorType::get(rt.getShape(), newElemType, rt.getEncoding());
  }

  return t;
}

class FloatToIntPass : public impl::FloatToIntBase<FloatToIntPass>
{
public:

  void runOnOperation() override
  {
    // Get current func::FuncOp
    auto F = getOperation();

    IRRewriter rewriter(F.getContext());

    // Update function signature
    for (BlockArgument arg : F.getArguments())
    {
      arg.setType(convertType(arg.getType()));
    }

    // Update function input and output type list
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;

    for (Type t : F.getArgumentTypes())
    {
      newInputs.push_back(convertType(t));
    }

    for (Type t : F.getResultTypes())
    {
      newResults.push_back(convertType(t));
    }

    // Override original type of the function
    F.setFunctionType(rewriter.getFunctionType(newInputs, newResults));

    // Collect the operations (AST walk)
    std::vector<Operation *> constantOpsToReplace;
    std::vector<Operation *> arithOpsToReplace;

    F.walk([&](Operation *op)
    {
      if (auto cst = llvm::dyn_cast<arith::ConstantOp>(op))
      {
        if (convertType(cst.getType()) != cst.getType())
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

      Type newType = convertType(op->getResult(0).getType());
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