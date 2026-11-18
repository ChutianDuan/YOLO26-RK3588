#pragma once
#include "camera/live_media.h"
namespace yolo::live {
void recordFixture(FrameSource& source, const std::filesystem::path& directory,
                   uint64_t frames, const std::atomic<bool>& stop);
}
