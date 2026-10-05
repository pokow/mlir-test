module
{
  func.func @rectangle_area_plus_offset(%width: i32, %height: i32, %offset: i32) -> i32
  {
    %area = arith.muli %width, %height : i32
    %result = arith.addi %area, %offset : i32
    return %result : i32
  }
}