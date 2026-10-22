#include "my/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"

using namespace mlir;

namespace my {

#define GEN_PASS_DEF_FLOATTOINT
#include "my/Transforms/Passes.h.inc"

namespace {

// Identifica operações estritamente restritas a ponto flutuante no TableGen do StableHLO
static bool isFloatOnlyMathOp(Operation *op) {
  return llvm::isa<stablehlo::ExpOp,
                   stablehlo::Expm1Op,
                   stablehlo::LogOp,
                   stablehlo::Log1pOp,
                   stablehlo::LogisticOp,
                   stablehlo::TanhOp,
                   stablehlo::SineOp,
                   stablehlo::CosineOp,
                   stablehlo::TanOp,
                   stablehlo::Atan2Op,
                   stablehlo::SqrtOp,
                   stablehlo::RsqrtOp,
                   stablehlo::CbrtOp,
                   stablehlo::CeilOp,
                   stablehlo::FloorOp,
                   stablehlo::RoundOp,
                   stablehlo::RoundNearestEvenOp,
                   stablehlo::IsFiniteOp,
                   stablehlo::ComplexOp,
                   stablehlo::RealOp,
                   stablehlo::ImagOp,
                   stablehlo::ReducePrecisionOp>(op);
}

// TypeConverter com suporte a i8 e Materializações Automáticas de Pontes (i8 <-> f32)
class FloatToIntTypeConverter : public TypeConverter {
public:
  FloatToIntTypeConverter() {
    addConversion([](Type t) -> Type { return t; });

    // Converte qualquer FloatType (f32, f16, etc.) para i8
    addConversion([](FloatType f) -> Type {
      return IntegerType::get(f.getContext(), 8);
    });

    addConversion([this](RankedTensorType rt) -> Type {
      Type elem = convertType(rt.getElementType());
      if (!elem)
        return nullptr;
      return RankedTensorType::get(rt.getShape(), elem, rt.getEncoding());
    });

    // Materialização de Origem: Injeta stablehlo.convert quando uma op legal (ex: exp) precisa de f32 a partir de i8
    addSourceMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      return stablehlo::ConvertOp::create(builder, loc, resultType, inputs[0]);
    });

    // Materialização de Destino: Injeta stablehlo.convert quando o resultado f32 deve alimentar uma op convertida para i8
    addTargetMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      return stablehlo::ConvertOp::create(builder, loc, resultType, inputs[0]);
    });
  }
};

// Conversão de Constantes por Truncamento (Float -> i8)
class ConvertConstantOp : public OpConversionPattern<stablehlo::ConstantOp> {
  using OpConversionPattern<stablehlo::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(stablehlo::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Attribute valueAttr = op.getValue();
    Type newResultType = getTypeConverter()->convertType(op.getType());
    if (!newResultType)
      return failure();

    ElementsAttr newValueAttr;

    if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(valueAttr)) {
      Type newElemType = llvm::cast<ShapedType>(newResultType).getElementType();
      unsigned bitWidth = newElemType.getIntOrFloatBitWidth();

      SmallVector<APInt> intValues;
      intValues.reserve(denseAttr.getNumElements());

      for (APFloat floatVal : denseAttr.getValues<APFloat>()) {
        APSInt intVal(bitWidth, /*isUnsigned=*/false);
        bool isExact;
        floatVal.convertToInteger(intVal, APFloat::rmTowardZero, &isExact);
        intValues.push_back(intVal);
      }

      newValueAttr = DenseElementsAttr::get(llvm::cast<ShapedType>(newResultType), intValues);
    } else if (auto resourceAttr = llvm::dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
      newValueAttr = DenseResourceElementsAttr::get(
          llvm::cast<ShapedType>(newResultType), resourceAttr.getRawHandle());
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, newResultType, newValueAttr);
    return success();
  }
};

// Conversão genérica de tipos para operações estruturais e aritméticas (add, conv, dot, etc.)
class GenericOpTypeConversion : public ConversionPattern {
public:
  GenericOpTypeConversion(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<func::FuncOp, stablehlo::ConstantOp>(op) || isFloatOnlyMathOp(op))
      return failure();

    SmallVector<Type, 4> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), newResultTypes)))
      return failure();

    OperationState state(op->getLoc(), op->getName(), operands,
                         newResultTypes, op->getAttrs(), op->getSuccessors());
    for (unsigned i = 0; i < op->getNumRegions(); ++i)
      state.addRegion();

    Operation *newOp = rewriter.create(state);
    for (auto [srcRegion, dstRegion] : llvm::zip(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *getTypeConverter())))
        return failure();
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class FloatToIntPass : public impl::FloatToIntBase<FloatToIntPass> {
public:
  using impl::FloatToIntBase<FloatToIntPass>::FloatToIntBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    FloatToIntTypeConverter typeConverter;
    RewritePatternSet patterns(ctx);

    // Padrões de reescrita de constantes e operações genéricas
    patterns.add<ConvertConstantOp>(typeConverter, ctx);
    patterns.add<GenericOpTypeConversion>(typeConverter, ctx);

    // Helpers padrão do MLIR para conversão do dialeto func (assinaturas, chamadas e retornos)
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(*ctx);

    // Regras de legalidade para funções e retornos
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getFunctionBody());
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op->getOperandTypes());
    });

    target.addDynamicallyLegalOp<stablehlo::ConstantOp>([&](stablehlo::ConstantOp op) {
      return typeConverter.isLegal(op.getType());
    });

    // Regras dinâmicas para o restante do IR
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      // 1. Operações exclusivamente float são LEGAIS se mantiverem os seus operandos e resultados em float
      if (isFloatOnlyMathOp(op)) {
        return llvm::any_of(op->getResultTypes(), [](Type t) {
          if (auto st = llvm::dyn_cast<ShapedType>(t))
            return llvm::isa<FloatType>(st.getElementType());
          return llvm::isa<FloatType>(t);
        });
      }

      // 2. A instrução stablehlo.convert é LEGAL para realizar a ponte de tipos (i8 <-> f32)
      if (isa<stablehlo::ConvertOp>(op))
        return true;

      // 3. As restantes operações devem estar convertidas para i8
      return typeConverter.isLegal(op);
    });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace my