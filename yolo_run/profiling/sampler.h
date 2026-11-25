#pragma once
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
namespace yolo::trace {
class Sampler {
  public:
    explicit Sampler(const std::filesystem::path &directory);
    void sample();
    bool good() const { return output_.good(); }

  private:
    std::ofstream output_;
    std::map<std::string, std::filesystem::path> gauges_;
    std::map<std::string, std::pair<uint64_t, uint64_t>> cpu_;
};
} // namespace yolo::trace
