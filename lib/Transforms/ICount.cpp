#include "my/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace my 
{

#define GEN_PASS_DEF_ICOUNT
#include "my/Transforms/Passes.h.inc"

namespace
{

class ICountPass : public impl::ICountBase<ICountPass>
{
public:
  void runOnOperation() override
  {
    int64_t count = 0;
    getOperation().walk([&](mlir::Operation *) { ++count; });
    llvm::outs() << "Instruction count: " << count << "\n";
  }
};

} // namespace
} // namespace my