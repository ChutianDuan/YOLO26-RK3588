#include "fixture_recorder.h"
#include <fstream>
namespace yolo::live {
using Json = nlohmann::json;
void recordFixture(FrameSource& source, const std::filesystem::path& directory, uint64_t frames, const std::atomic<bool>& stop) {
    if (!frames || !std::filesystem::create_directory(directory)) throw std::runtime_error("fixture directory must be new; frame count positive");
    std::ofstream manifest(directory / "manifest.jsonl");
    auto meta = source.info(); meta["type"] = "mjpeg_fixture";
    manifest << meta.dump() << '\n';
    for (uint64_t i = 0; i < frames; ++i) {
        auto packet = source.next(stop); if (!packet) break;
        const auto name = std::to_string(i) + ".jpg";
        std::ofstream file(directory / name, std::ios::binary);
        packet->withCpuRead([&](const unsigned char* data, size_t bytes) {
            file.write(reinterpret_cast<const char*>(data), bytes);
        });
        if (!file) throw std::runtime_error("fixture JPEG write failed");
        const auto& s = packet->stamp;
        auto row = nlohmann::json{{"sequence", s.sequence}, {"timestamp_ms", s.timestamp_ms},
            {"raw_timestamp_ms", s.raw_timestamp_ms}, {"timestamp_flags", s.flags}, {"timestamp_clock", s.clock}}; row["file"] = name;
        manifest << row.dump() << '\n';
        if (!manifest) throw std::runtime_error("fixture manifest write failed");
    }
}

}
