#pragma once
#include <rknn_api.h>
#include "config/app_config.h"
#include "model/inference_types.h"

namespace yolo {
// Only the deployed YOLO26 one-to-one, two-output export is accepted.
void validateModelLayout(const rknn_tensor_attr& input,
                         const std::vector<rknn_tensor_attr>& outputs,
                         const AppConfig& config);
std::vector<Detection> decodeYolo26(const float* boxes, const float* scores,
                                  size_t anchors, const TensorInput& input,
                                  const AppConfig& config);
}  // namespace yolo
