#include "config/app_config.h"
#include "image/dma_buffer.h"
#include "../support/cpu_reference.h"
#include "image/image_processing.h"
#include "model/yolo_engine.h"
#include "model/yolo_postprocess.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
using namespace yolo;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}
AppConfig cocoConfig() {
    return loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config/camera_coco640.yaml");
}
rknn_tensor_attr inputAttribute(const AppConfig& c) {
    rknn_tensor_attr input{};
    input.n_dims = 4;
    input.dims[0] = 1; input.dims[1] = c.input_height;
    input.dims[2] = c.input_width; input.dims[3] = 3;
    input.n_elems = c.input_width * c.input_height * 3;
    input.fmt = RKNN_TENSOR_NHWC; input.type = RKNN_TENSOR_FLOAT16;
    return input;
}
std::vector<rknn_tensor_attr> outputAttributes(int classes, int candidates) {
    std::vector<rknn_tensor_attr> outputs(2);
    for (int i = 0; i < 2; ++i) {
        auto& a = outputs[i];
        a.index = i; a.n_dims = 3; a.dims[0] = 1;
        a.dims[1] = i ? classes : 4; a.dims[2] = candidates;
        a.n_elems = a.dims[1] * a.dims[2];
        a.fmt = RKNN_TENSOR_UNDEFINED; a.type = RKNN_TENSOR_FLOAT16;
    }
    return outputs;
}
void coreChecks() {
    auto c = cocoConfig();
    require(c.input_width == 640 && c.input_height == 640 && c.num_classes == 80,
            "COCO config dimensions/classes");
    require(c.class_names.size() == 80 && c.class_names.front() == "person" &&
            c.class_names[1] == "bicycle" && c.class_names[5] == "bus" &&
            c.class_names.back() == "toothbrush", "COCO label order");
    auto changed = c;
    changed.class_conf_thresholds.assign(10, -1);
    rejects([&] { validateConfig(changed); }, "legacy threshold count accepted for COCO");
    changed.class_conf_thresholds.assign(80, -1);
    validateConfig(changed);
    changed.class_names.pop_back();
    rejects([&] { validateConfig(changed); }, "incomplete COCO labels accepted");

    auto input = inputAttribute(c);
    auto outputs = outputAttributes(80, 8400);
    validateModelLayout(input, outputs, c);
    std::swap(outputs[0], outputs[1]);
    validateModelLayout(input, outputs, c);
    outputs[0].dims[2] = 8399; outputs[0].n_elems = 80 * 8399;
    rejects([&] { validateModelLayout(input, outputs, c); }, "mismatched candidate counts accepted");
    outputs = outputAttributes(80, 8400);
    outputs[1].dims[1] = 10; outputs[1].n_elems = 10 * 8400;
    rejects([&] { validateModelLayout(input, outputs, c); }, "legacy classes accepted for COCO");
    outputs = outputAttributes(80, 8400);
    input.fmt = RKNN_TENSOR_NCHW;
    rejects([&] { validateModelLayout(input, outputs, c); }, "wrong RGB tensor layout accepted");
    const auto legacy = loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config.yaml");
    validateModelLayout(inputAttribute(legacy), outputAttributes(10, 9975), legacy);

    const auto g = resizeGeometry(800, 600, c);
    require(g.width == 640 && g.height == 480 && g.left == 0 && g.top == 80 &&
            std::abs(g.transform.scale_x - .8F) < 1e-6F &&
            std::abs(g.transform.scale_y - .8F) < 1e-6F,
            "800x600 letterbox geometry");
    TensorInput tensor;
    tensor.image_width = 800; tensor.image_height = 600; tensor.letterbox = g.transform;
    // First box maps to (100,100,300,400), second clips to the source boundaries,
    // and the third lies entirely in the top padding and must disappear.
    const float boxes[]{80,-8,0, 160,64,0, 240,648,80, 400,576,64};
    std::vector<float> scores(80 * 3, 0);
    scores[0] = .95F; scores[79 * 3 + 1] = .9F; scores[3 + 2] = .85F;
    auto detections = decodeYolo26(boxes, scores.data(), 3, tensor, c);
    require(detections.size() == 2 && detections[0].class_id == 0 &&
            std::abs(detections[0].x1 - 100) < 1e-4F &&
            std::abs(detections[0].y1 - 100) < 1e-4F &&
            std::abs(detections[0].x2 - 300) < 1e-4F &&
            std::abs(detections[0].y2 - 400) < 1e-4F &&
            detections[1].class_id == 79 && detections[1].x1 == 0 &&
            detections[1].y1 == 0 && detections[1].x2 == 800 && detections[1].y2 == 600,
            "letterbox inverse, clipping or padded detection filtering");
    std::fill(scores.begin(), scores.end(), 0);
    scores[0] = .95F; scores[3] = .94F; scores[1] = .93F;
    detections = decodeYolo26(boxes, scores.data(), 3, tensor, c);
    require(detections.size() == 3 && detections[0].class_id == 0 &&
            detections[1].class_id == 1 && detections[0].x1 == detections[1].x1 &&
            detections[0].y1 == detections[1].y1,
            "one2one TopK must retain multiple classes on one candidate without NMS");
    c.class_conf_thresholds.assign(80, -1); c.class_conf_thresholds[0] = .96F;
    require(decodeYolo26(boxes, scores.data(), 3, tensor, c).size() == 1,
            "COCO per-class threshold not applied");
    c.class_conf_thresholds.clear();
    constexpr size_t count = 350;
    std::vector<float> many_boxes(count * 4), many_scores(count * 80, 0);
    for (size_t i = 0; i < count; ++i) {
        many_boxes[i] = 80; many_boxes[count+i] = 160;
        many_boxes[2*count+i] = 240; many_boxes[3*count+i] = 400;
        many_scores[79*count+i] = .5F + float(i) / 1000;
    }
    detections = decodeYolo26(many_boxes.data(), many_scores.data(), count, tensor, c);
    require(detections.size() == 300 && detections.front().class_id == 79 &&
            std::abs(detections.front().score - .849F) < 1e-6F &&
            std::abs(detections.back().score - .55F) < 1e-6F,
            "COCO TopK limit/order or last-class indexing");
    std::cout << "PASS COCO config, split tensor contract, legacy layout, letterbox and NMS-free TopK\n";
}
cv::Mat snapshotRgb(const TensorInput& input) {
    require(input.dma.allocation != nullptr, "expected RGB DMA input");
    auto allocation = std::static_pointer_cast<DmaBuffer>(input.dma.allocation);
    CpuAccess read(allocation, DMA_BUF_SYNC_READ);
    cv::Mat result = cv::Mat(input.dma.height, input.dma.width, CV_8UC3,
                            input.dma.data, size_t(input.dma.width_stride) * 3).clone();
    read.finish();
    return result;
}
void rgaChecks(const AppConfig& c) {
    constexpr int stride = 832, height_stride = 608;
    auto allocation = std::make_shared<DmaBuffer>(size_t(stride) * height_stride * 3 / 2,
                                                "/dev/dma_heap/system-uncached");
    auto lease = std::make_shared<int>(1);
    Nv12FrameView frame{lease, allocation, 800, 600, stride, height_stride, true};
    const std::array<std::array<unsigned char,3>,5> yuv{{
        {0,128,128}, {255,128,128}, {76,85,255}, {150,44,21}, {29,255,107}}};
    const std::array<cv::Vec3b,5> expected{{{0,0,0}, {255,255,255}, {255,0,0}, {0,255,0}, {0,0,255}}};
    ImagePreprocessor processor(c, {656, 642, 0});
    int previous_fd = -1;
    for (int iteration = 0; iteration < 3; ++iteration) {
        {
            CpuAccess write(allocation, DMA_BUF_SYNC_WRITE);
            std::memset(allocation->data, 0xCD, allocation->size);
            auto* data = static_cast<unsigned char*>(allocation->data);
            for (int y = 0; y < 600; ++y) for (int x = 0; x < 800; ++x)
                data[size_t(y)*stride+x] = yuv[(x/160+iteration)%5][0];
            auto* uv = data + size_t(stride) * height_stride;
            for (int y = 0; y < 300; ++y) for (int x = 0; x < 800; x += 2) {
                const auto& color = yuv[(x/160+iteration)%5];
                uv[size_t(y)*stride+x] = color[1]; uv[size_t(y)*stride+x+1] = color[2];
            }
            write.finish();
        }
        auto tensor = processor.prepareNv12(frame);
        require(tensor.dma.width == 640 && tensor.dma.height == 640 &&
                tensor.dma.width_stride == 656 && tensor.dma.height_stride == 642,
                "COCO RGB destination strides");
        if (previous_fd >= 0) require(tensor.dma.fd == previous_fd, "COCO RGB allocation not reused");
        previous_fd = tensor.dma.fd;
        const auto pixels = snapshotRgb(tensor);
        require(cv::norm(pixels.rowRange(0,80), cv::Mat(80,640,CV_8UC3,cv::Scalar::all(114)), cv::NORM_INF) == 0 &&
                cv::norm(pixels.rowRange(560,640), cv::Mat(80,640,CV_8UC3,cv::Scalar::all(114)), cv::NORM_INF) == 0,
                "RGA COCO padding is not 114");
        for (int i = 0; i < 5; ++i) {
            const auto actual = pixels.at<cv::Vec3b>(320, i*128+64);
            const auto reference = expected[(i+iteration)%5];
            for (int channel = 0; channel < 3; ++channel)
                require(std::abs(int(actual[channel])-int(reference[channel])) <= 4,
                        "NV12 full-range BT.601/RGB/stride conversion");
        }
        rejects([&] { processor.prepareNv12(frame); }, "leased COCO RGB buffer overwritten");
        auto alias = tensor; tensor = {};
        rejects([&] { processor.prepareNv12(frame); }, "copied COCO RGB lease ignored");
        alias = {};
    }
    std::cout << "PASS RGA NV12 800x600 to RGB 640x640, padding, colors, strides and reuse leases\n";
}
void equivalent(const InferResult& a, const InferResult& b) {
    require(a.detections.size() == b.detections.size(), "same RGB CPU/DMA detection count mismatch");
    for (size_t i = 0; i < a.detections.size(); ++i) {
        const auto& x = a.detections[i]; const auto& y = b.detections[i];
        require(x.class_id == y.class_id && std::abs(x.score-y.score) <= 1e-6F &&
                std::abs(x.x1-y.x1) <= 1e-3F && std::abs(x.y1-y.y1) <= 1e-3F &&
                std::abs(x.x2-y.x2) <= 1e-3F && std::abs(x.y2-y.y2) <= 1e-3F,
                "same RGB CPU/DMA class, ordering, score or box mismatch");
    }
}
void hardwareChecks() {
    const auto c = cocoConfig();
    rgaChecks(c);
    cv::Mat bus = cv::imread("/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg"), image;
    require(!bus.empty(), "missing installed bus.jpg fixture");
    cv::resize(bus, image, {800,600});
    YoloEngine engine(c);
    ImagePreprocessor direct(c, engine.inputMemoryLayout()), reference(c, engine.inputMemoryLayout());
    auto dma_tensor = direct.prepare(image);
    auto reference_tensor = reference.prepare(image);
    auto copied_pixels = std::make_shared<cv::Mat>(snapshotRgb(reference_tensor));
    yolo::test::CpuReferenceEngine cpu_reference(c);
    // Do not open CPU access on the direct RGB allocation before FD inference;
    // doing so would hide device-to-device cache/synchronization failures.
    const auto cpu = cpu_reference.infer(*copied_pixels, dma_tensor);
    const auto dma = engine.infer(dma_tensor);
    const auto repeated = engine.infer(dma_tensor);
    equivalent(cpu, dma); equivalent(dma, repeated);
    require(cv::norm(snapshotRgb(dma_tensor), *copied_pixels, cv::NORM_INF) == 0,
            "same-input inference comparison used different RGB pixels");
    bool person = false, vehicle = false;
    require(!dma.detections.empty() && dma.detections.size() <= 300,
            "COCO FP16 bus fixture returned no detections or exceeded TopK");
    for (const auto& d : dma.detections) {
        require(d.class_id >= 0 && d.class_id < 80 && std::isfinite(d.score) &&
                std::isfinite(d.x1) && std::isfinite(d.y1) && std::isfinite(d.x2) && std::isfinite(d.y2) &&
                d.score >= c.conf_threshold && d.score <= 1 && d.x1 >= 0 && d.y1 >= 0 &&
                d.x2 <= 800 && d.y2 <= 600 && d.x2 > d.x1 && d.y2 > d.y1,
                "invalid COCO FP16 detection in source coordinates");
        person |= d.class_id == 0; vehicle |= d.class_id == 5;
    }
    require(person && vehicle, "COCO bus fixture did not detect both person and bus");
    std::cout << "PASS COCO FP16 RKNN CPU/DMA/repeat equivalence: " << dma.detections.size()
              << " detections, CPU infer=" << cpu.infer_ms << "ms, DMA infer=" << dma.infer_ms << "ms\n";
}
}
int main(int argc, char** argv) {
    try {
        cv::setNumThreads(1);
        if (argc == 1) coreChecks();
        else if (argc == 2 && std::string(argv[1]) == "--hardware") hardwareChecks();
        else throw std::runtime_error("Usage: coco_model_test [--hardware]");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
