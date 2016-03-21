require "test/unit"
require "cudnn"

class TestCuDNN < Test::Unit::TestCase
  def test_s
    Cudnn::Handle.new
    v = Cudnn::DeviceVectorFloat.new([1,1,1])
    assert_equal([1,1,1], v.to_hostvec)
  end
end
