module
{
  func.func @square(%x: tensor<4xf32>) -> tensor<4xf32>
  {
    %0 = arith.mulf %x, %x : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @main(%a: tensor<4xf32>) -> tensor<4xf32>
  {
    %c = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %0 = arith.addf %a, %c : tensor<4xf32>
    %1 = func.call @square(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}