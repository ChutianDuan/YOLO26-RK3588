#include "model/yolo_postprocess.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace yolo {
namespace {
void validateQuantization(const rknn_tensor_attr& a) {
    if (a.type != RKNN_TENSOR_FLOAT16 && a.type != RKNN_TENSOR_FLOAT32 && a.type != RKNN_TENSOR_INT8)
        throw std::runtime_error("unsupported tensor type");
    if (a.type == RKNN_TENSOR_INT8 && (a.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
        || !std::isfinite(a.scale) || a.scale <= 0 || a.zp < -128 || a.zp > 127))
        throw std::runtime_error("invalid INT8 affine quantization");
}
}
void validateModelLayout(const rknn_tensor_attr& in, const std::vector<rknn_tensor_attr>& out, const AppConfig& c) {
    // 解码后的 one2one 导出契约：RGB NHWC、boxes[1,4,N] / scores[1,C,N]。
    // 先验证维度、类型和量化参数，避免形状近似的其他 YOLO 模型被误解码。
    if (in.n_dims != 4 || in.fmt != RKNN_TENSOR_NHWC || in.dims[0] != 1 ||
        in.dims[1] != unsigned(c.input_height) || in.dims[2] != unsigned(c.input_width) || in.dims[3] != 3 ||
        in.n_elems != unsigned(c.input_width * c.input_height * 3))
        throw std::runtime_error("RKNN input must be [1," + std::to_string(c.input_height) + "," +
                                 std::to_string(c.input_width) + ",3] NHWC");
    validateQuantization(in);
    if (out.size() != 2) throw std::runtime_error("YOLO26 requires exactly boxes and scores outputs");
    bool boxes = false, scores = false;
    uint32_t anchors = 0;
    for (const auto& a : out) {
        validateQuantization(a);
        if (a.n_dims != 3 || a.dims[0] != 1 || !a.dims[2] ||
            uint64_t(a.n_elems) != uint64_t(a.dims[1]) * a.dims[2] ||
            a.n_elems > std::numeric_limits<uint32_t>::max() / sizeof(float) ||
            (a.fmt != RKNN_TENSOR_NCHW && a.fmt != RKNN_TENSOR_UNDEFINED))
            throw std::runtime_error("unexpected YOLO26 output layout");
        if (anchors && anchors != a.dims[2]) throw std::runtime_error("YOLO26 outputs have different candidate counts");
        anchors = a.dims[2];
        // 四类模型的两个输出具有相同形状，必须用约定的名称消除歧义。
        const bool box_tensor = a.dims[1] == 4 &&
            (c.num_classes != 4 || std::strncmp(a.name, "boxes", sizeof(a.name)) == 0);
        const bool score_tensor = a.dims[1] == unsigned(c.num_classes) &&
            (c.num_classes != 4 || std::strncmp(a.name, "scores", sizeof(a.name)) == 0);
        if (box_tensor && !boxes) boxes = true;
        else if (score_tensor && !scores) scores = true;
        else throw std::runtime_error("expected boxes[1,4,N] and scores[1," + std::to_string(c.num_classes) + ",N]");
    }
    if (!boxes || !scores) throw std::runtime_error("missing YOLO26 output");
}

std::vector<Detection> decodeYolo26(const float* boxes, const float* scores, size_t anchors,
                                   const TensorInput& input, const AppConfig& c) {
    if (!boxes || !scores || anchors == 0 || c.num_classes <= 0 ||
        (!c.class_conf_thresholds.empty() && c.class_conf_thresholds.size() != size_t(c.num_classes)) ||
        input.image_width <= 0 || input.image_height <= 0 ||
        !std::isfinite(input.letterbox.scale_x) || !std::isfinite(input.letterbox.scale_y) ||
        input.letterbox.scale_x <= 0 || input.letterbox.scale_y <= 0)
        throw std::runtime_error("invalid YOLO26 decoder input");
    // 对齐当前 YOLO26 one-to-one 导出：先按每个候选的最高类别分数选 TopK，
    // 再对候选/类别组合选 TopK；同一候选可保留多个类别，不额外执行 NMS。
    // 并列分数按原始次序排序，与参考 NumPy 稳定排序一致，尤其影响 INT8 大量同分情况。
    struct Anchor { size_t index; float score; };
    std::vector<Anchor> ranked;
    ranked.reserve(anchors);
    for (size_t i = 0; i < anchors; ++i) {
        float score = -1;
        for (int j = 0; j < c.num_classes; ++j) {
            const float value = scores[size_t(j) * anchors + i];
            if (std::isfinite(value) && value >= 0 && value <= 1) score = std::max(score, value);
        }
        ranked.push_back({i, score});
    }
    const size_t count = std::min<size_t>(300, anchors);
    const auto anchorOrder = [](const Anchor& a, const Anchor& b) {
        return a.score != b.score ? a.score > b.score : a.index < b.index;
    };
    std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end(), anchorOrder);
    struct Pair { size_t anchor, order; int cls; float score; };
    std::vector<Pair> pairs;
    pairs.reserve(count * c.num_classes);
    for (size_t rank = 0; rank < count; ++rank) {
        for (int cls = 0; cls < c.num_classes; ++cls) {
            const float value = scores[size_t(cls) * anchors + ranked[rank].index];
            pairs.push_back({ranked[rank].index, rank * c.num_classes + size_t(cls), cls,
                std::isfinite(value) && value >= 0 && value <= 1 ? value : -1.0F});
        }
    }
    const auto pairOrder = [](const Pair& a, const Pair& b) {
        return a.score != b.score ? a.score > b.score : a.order < b.order;
    };
    std::partial_sort(pairs.begin(), pairs.begin() + count, pairs.end(), pairOrder);
    std::vector<Detection> detections;
    detections.reserve(count);
    for (size_t rank = 0; rank < count; ++rank) {
        const auto& pair = pairs[rank];
        const size_t i = pair.anchor;
        const int cls = pair.cls;
        const float score = pair.score;
        const float threshold = !c.class_conf_thresholds.empty() && c.class_conf_thresholds[cls] >= 0
            ? c.class_conf_thresholds[cls] : c.conf_threshold;
        if (score < threshold) continue;
        // Runtime 输出已是 FP32，不能再次按 scale/zp 手工反量化。
        float xy[4];
        bool finite = true;
        for (size_t j = 0; j < 4; ++j) { xy[j] = boxes[j * anchors + i]; finite &= std::isfinite(xy[j]); }
        if (!finite) continue;
        Detection d;
        d.class_id = cls;
        d.score = score;
        // 先减去 letterbox 左/上填充，再分别除以实际 X/Y 比例，恢复到原图像素坐标。
        // 限制到原图边界，并在下方剔除退化框；原始 BGR 图像不需要反向缩放。
        d.x1 = std::clamp((xy[0] - input.letterbox.pad_w) / input.letterbox.scale_x, 0.0F, float(input.image_width));
        d.y1 = std::clamp((xy[1] - input.letterbox.pad_h) / input.letterbox.scale_y, 0.0F, float(input.image_height));
        d.x2 = std::clamp((xy[2] - input.letterbox.pad_w) / input.letterbox.scale_x, 0.0F, float(input.image_width));
        d.y2 = std::clamp((xy[3] - input.letterbox.pad_h) / input.letterbox.scale_y, 0.0F, float(input.image_height));
        if (d.x2 > d.x1 && d.y2 > d.y1) detections.push_back(d);
    }
    return detections;
}
}  // namespace yolo
