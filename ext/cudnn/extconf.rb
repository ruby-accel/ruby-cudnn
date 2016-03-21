require 'mkmf'

have_library("c++") or have_library("stdc++")

cuda_lib_path = " -lcudart "
if cuda_config = dir_config("cuda")
  if /darwin/ =~ RbConfig::CONFIG["target_os"]
    cuda_lib_path = " -rpath #{cuda_config[1]} " + cuda_lib_path
  elsif /linux/ =~ RbConfig::CONFIG["target_os"]
    cuda_lib_path = " -L #{cuda_config[1]} " + cuda_lib_path
  end
end

cudnn_lib_path = " -lcudnn "
if cudnn_config = dir_config("cudnn")
  if /darwin/ =~ RbConfig::CONFIG["target_os"]
    cudnn_lib_path = " -rpath #{cudnn_config[1]} " + cudnn_lib_path
  elsif /linux/ =~ RbConfig::CONFIG["target_os"]
    cudnn_lib_path = " -L #{cudnn_config[1]} " + cudnn_lib_path
  end
end

$CXXFLAGS = ($CXXFLAGS || "") + " -O2 -Wall "
$LDFLAGS = ($LDFLAGS || "") + cuda_lib_path + cudnn_lib_path
create_makefile('cudnn/cudnn')
