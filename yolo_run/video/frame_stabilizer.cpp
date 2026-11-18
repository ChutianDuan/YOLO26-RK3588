#include "video/frame_stabilizer.h"
#include "image/dma_buffer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace yolo {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
float median(std::vector<float>& values) {
    const size_t middle = values.size()/2;
    std::nth_element(values.begin(), values.begin()+middle, values.end());
    if (values.size()%2) return values[middle];
    return (values[middle] + *std::max_element(values.begin(), values.begin()+middle))/2;
}
void checkRga(IM_STATUS status, const char* operation) {
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR)
        throw std::runtime_error(std::string(operation)+": "+imStrError(status));
}
}

GrayStabilizer::GrayStabilizer(double alpha) : alpha_(alpha) {
    if (!std::isfinite(alpha) || alpha <= 0 || alpha > 1)
        throw std::invalid_argument("stabilize-alpha must be in (0,1]");
    for (auto* values : {&points_, &candidates_, &forward_, &backward_}) values->reserve(192);
    for (auto* values : {&error_, &backward_error_, &dx_, &dy_, &scratch_}) values->reserve(192);
    status_.reserve(192); backward_status_.reserve(192); accepted_.reserve(192);
}

StabilizationInfo GrayStabilizer::update(const cv::Mat& gray, double timestamp,
                                        const std::vector<TrackedDetection>& tracks) {
    if (gray.type()!=CV_8UC1 || gray.size()!=cv::Size(320,240) || !std::isfinite(timestamp))
        throw std::invalid_argument("stabilization requires 320x240 gray and a finite timestamp");
    const auto start = Clock::now();
    StabilizationInfo info;
    info.enabled = true;
    const double dt = timestamp-timestamp_;
    info.reference_reset = previous_.empty() || dt <= 0 || dt > 250;
    cv::buildOpticalFlowPyramid(gray, pyramids_[1], {21,21}, 2, true);
    if (!info.reference_reset) {
        mask_.create(gray.size(), CV_8UC1); mask_.setTo(255);
        // Tracks live on the stabilized canvas at the previous image timestamp.
        // Invert that frame's actual (aligned) ROI before masking raw-image features.
        for (const auto& track : tracks) {
            const auto& box = track.detection;
            const double grow_x=(box.x2-box.x1)*0.2, grow_y=(box.y2-box.y1)*0.2;
            const int x1=std::clamp(int(std::floor((roi_.x+(box.x1-grow_x)*0.9)*0.4)),0,320);
            const int y1=std::clamp(int(std::floor((roi_.y+(box.y1-grow_y)*0.9)*0.4)),0,240);
            const int x2=std::clamp(int(std::ceil((roi_.x+(box.x2+grow_x)*0.9)*0.4)),0,320);
            const int y2=std::clamp(int(std::ceil((roi_.y+(box.y2+grow_y)*0.9)*0.4)),0,240);
            if (x2>x1 && y2>y1) mask_(cv::Rect(x1,y1,x2-x1,y2-y1)).setTo(0);
        }
        points_.clear();
        for (int row=0; row<3; ++row) for (int col=0; col<4; ++col) {
            const cv::Rect cell(col*80,row*80,80,80);
            cv::goodFeaturesToTrack(previous_(cell), candidates_, 16, 0.01, 8, mask_(cell));
            for (auto p : candidates_) points_.emplace_back(p.x+cell.x,p.y+cell.y);
        }
        info.sampled_points = int(points_.size());
        dx_.clear(); dy_.clear(); accepted_.clear();
        if (points_.size()>=12) {
            cv::calcOpticalFlowPyrLK(pyramids_[0],pyramids_[1],points_,forward_,status_,error_,{21,21},2,
                {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,12,0.03});
            cv::calcOpticalFlowPyrLK(pyramids_[1],pyramids_[0],forward_,backward_,backward_status_,backward_error_,{21,21},2,
                {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,12,0.03});
            for (size_t i=0; i<points_.size(); ++i) {
                const auto p=forward_[i], back=backward_[i];
                if (!status_[i] || !backward_status_[i] || !std::isfinite(p.x) || !std::isfinite(p.y) ||
                    !std::isfinite(back.x) || !std::isfinite(back.y) || p.x<0 || p.x>=320 || p.y<0 || p.y>=240 ||
                    cv::norm(points_[i]-back)>1.5) continue;
                dx_.push_back(p.x-points_[i].x); dy_.push_back(p.y-points_[i].y); accepted_.push_back(i);
            }
            info.valid_points=int(dx_.size());
            if (dx_.size()>=12) {
                scratch_=dx_; const double x=median(scratch_);
                scratch_=dy_; const double y=median(scratch_);
                std::array<bool,12> cells{};
                unsigned rows=0, columns=0;
                for (size_t i=0; i<dx_.size(); ++i) if (std::hypot(dx_[i]-x,dy_[i]-y)<=1.5) {
                    ++info.inliers;
                    const auto p=points_[accepted_[i]];
                    const int col=std::clamp(int(p.x)/80,0,3), row=std::clamp(int(p.y)/80,0,2);
                    cells[row*4+col]=true; rows|=1U<<row; columns|=1U<<col;
                }
                info.background_cells=int(std::count(cells.begin(),cells.end(),true));
                info.inlier_ratio=double(info.inliers)/dx_.size();
                const bool spread=(rows&(rows-1)) && (columns&(columns-1));
                info.valid=info.inliers>=12 && info.inlier_ratio>=0.65 && info.background_cells>=4 && spread;
                if (info.valid) info.delta={x*2.5,y*2.5};
            }
        }
    }
    // Exact equivalent of cumulative trajectory minus its EMA; no unbounded integral.
    // Invalid estimates contribute zero motion and ease the current offset back to center.
    const double alpha_dt=1-std::pow(1-alpha_,info.reference_reset ? 1 : dt/(1000.0/30));
    compensation_=(compensation_+info.delta)*(1-alpha_dt);
    const cv::Point2d limited(std::clamp(compensation_.x,-40.0,40.0),std::clamp(compensation_.y,-30.0,30.0));
    info.clamped=limited!=compensation_;
    compensation_=limited;
    roi_={std::clamp(int(2*std::lround((40+limited.x)/2)),0,80),
          std::clamp(int(2*std::lround((30+limited.y)/2)),0,60),720,540};
    info.roi=roi_; info.compensation=compensation_;
    previous_=gray; timestamp_=timestamp; pyramids_[0].swap(pyramids_[1]);
    info.motion_ms=ms(start); info.total_ms=info.motion_ms;
    return info;
}

class FrameStabilizer::Impl {
public:
    explicit Impl(double alpha) : estimator(alpha) {}
    struct Slot { std::shared_ptr<DmaBuffer> dma; std::unique_ptr<CpuAccess> read; };
    StabilizationInfo update(const Nv12FrameView& frame, double timestamp,
                             const std::vector<TrackedDetection>& tracks) {
        const auto start=Clock::now();
        const auto region=nv12SourceRegion(frame);
        if (region!=cv::Rect(0,0,800,600)) throw std::invalid_argument("stabilization expects 800x600 NV12");
        auto& slot=slots[next];
        if (slot.read) { slot.read->finish(); slot.read.reset(); }
        if (!slot.dma) slot.dma=std::make_shared<DmaBuffer>(320U*240U*3/2);
        if (!slot.dma->healthy) throw std::runtime_error("stabilization gray DMA sync failure");
        auto src=wrapbuffer_handle(frame.dma->handle,800,600,RK_FORMAT_YCbCr_420_SP,frame.width_stride,frame.height_stride);
        auto dst=wrapbuffer_handle(slot.dma->handle,320,240,RK_FORMAT_YCbCr_420_SP,320,240);
        checkRga(imcheck(src,dst,{}, {},0),"stabilization gray layout");
        im_opt_t options{}; options.core=IM_SCHEDULER_RGA3_CORE0|IM_SCHEDULER_RGA3_CORE1;
        const auto status=improcess(src,dst,{}, {},{}, {},-1,nullptr,&options,IM_SYNC);
        if (status!=IM_STATUS_SUCCESS && status!=IM_STATUS_NOERROR) {
            std::cerr<<"fatal: stabilization RGA completion uncertain: "<<imStrError(status)<<'\n'; std::terminate();
        }
        slot.read=std::make_unique<CpuAccess>(slot.dma,DMA_BUF_SYNC_READ);
        const double gray_ms=ms(start);
        auto info=estimator.update(cv::Mat(240,320,CV_8UC1,slot.dma->data,320),timestamp,tracks);
        // The former previous gray is no longer read by LK; its slot may be overwritten next time.
        auto& old=slots[1-next];
        if (old.read) { old.read->finish(); old.read.reset(); }
        next=1-next; info.gray_ms=gray_ms; info.total_ms=ms(start);
        return info;
    }
    std::array<Slot,2> slots;
    size_t next=0;
    GrayStabilizer estimator;
};
FrameStabilizer::FrameStabilizer(double alpha) : impl_(std::make_unique<Impl>(alpha)) {}
FrameStabilizer::~FrameStabilizer()=default;
double FrameStabilizer::previousTimestamp() const { return impl_->estimator.previousTimestamp(); }
void FrameStabilizer::finish() {
    for (auto& slot:impl_->slots) if (slot.read) { slot.read->finish(); slot.read.reset(); }
}
StabilizationInfo FrameStabilizer::update(const Nv12FrameView& frame,double timestamp,
                                        const std::vector<TrackedDetection>& tracks) {
    return impl_->update(frame,timestamp,tracks);
}
} // namespace yolo
