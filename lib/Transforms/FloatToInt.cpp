#include "my/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

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

// TypeConverter: fN -> iN, tensor<...xfN> -> tensor<...xiN>
class FloatToIntTypeConverter : public TypeConverter {
public:
  FloatToIntTypeConverter() {
    addConversion([](Type t) -> Type { return t; });

    addConversion([](FloatType f) -> Type {
      return IntegerType::get(f.getContext(), f.getWidth());
    });

    addConversion([this](RankedTensorType rt) -> Type {
      Type elem = convertType(rt.getElementType());
      if (!elem)
        return nullptr;
      return RankedTensorType::get(rt.getShape(), elem, rt.getEncoding());
    });
  }
};

// Conversão de Constantes (Suporta DenseElementsAttr e DenseResourceElementsAttr)
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
      newValueAttr = denseAttr.bitcast(newElemType);
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

// Conversão de Assinatura de Funções
class ConvertFuncSignature : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    TypeConverter::SignatureConversion signatureConverter(funcOp.getNumArguments());

    if (failed(converter.convertSignatureArgs(funcOp.getFunctionType().getInputs(), signatureConverter)))
      return failure();

    SmallVector<Type, 4> newResultTypes;
    if (failed(converter.convertTypes(funcOp.getFunctionType().getResults(), newResultTypes)))
      return failure();

    auto newFuncType = rewriter.getFunctionType(signatureConverter.getConvertedTypes(), newResultTypes);

    rewriter.modifyOpInPlace(funcOp, [&]() {
      funcOp.setType(newFuncType);
      if (!funcOp.getBody().empty()) {
        rewriter.applySignatureConversion(&funcOp.getBody().front(), signatureConverter, &converter);
      }
    });

    return success();
  }
};

// Padrão 1 (Prioridade 2): Isola operações float-only em custom_call
class ConvertFloatMathOp : public ConversionPattern {
public:
  ConvertFloatMathOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/2, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isFloatOnlyMathOp(op))
      return failure();

    SmallVector<Type, 4> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), newResultTypes)))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op,
        newResultTypes,
        operands,
        rewriter.getStringAttr(op->getName().stripDialect()),
        /*has_side_effect=*/BoolAttr(),
        /*backend_config=*/Attribute(),
        /*api_version=*/stablehlo::CustomCallApiVersionAttr(),
        /*called_computations=*/ArrayAttr(),
        /*operand_layouts=*/ArrayAttr(),
        /*result_layouts=*/ArrayAttr(),
        /*output_operand_aliases=*/ArrayAttr());

    return success();
  }
};

// Padrão 2 (Prioridade 1): Atualiza o tipo de operações polimórficas/estruturais mantendo a instrução nativa
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

    patterns.add<ConvertConstantOp, ConvertFuncSignature>(typeConverter, ctx);
    patterns.add<ConvertFloatMathOp>(typeConverter, ctx);
    patterns.add<GenericOpTypeConversion>(typeConverter, ctx);

    ConversionTarget target(*ctx);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getFunctionBody());
    });

    target.addDynamicallyLegalOp<stablehlo::ConstantOp>([&](stablehlo::ConstantOp op) {
      return typeConverter.isLegal(op.getType());
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return typeConverter.isLegal(op);
    });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace my