#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  std::string arch(prop.gcnArchName);
  std::cout << "GPU: " << arch << std::endl;

  bool is_supported = (arch.find("gfx94") != std::string::npos) ||
                      (arch.find("gfx950") != std::string::npos) ||
                      (arch.find("gfx12") != std::string::npos);

  if (!is_supported) {
    std::cout << "Standard E4M3 NOT supported in kernels on this GPU."
              << std::endl;
    std::cout << "Only FNUZ (bias=8) emulation available." << std::endl;
    std::cout << "Our manual bias=7 code is required." << std::endl;
  } else {
    std::cout << "Standard E4M3 IS supported!" << std::endl;
  }
  return 0;
}
