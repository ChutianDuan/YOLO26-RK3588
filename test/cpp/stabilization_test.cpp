#include "video/frame_stabilizer.h"
#include "camera/live_media.h"
#include "image/dma_buffer.h"
#include "model/yolo_engine.h"
#include "model/yolo_postprocess.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {
using namespace yolo;
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool rejected=false; try { f(); } catch (const std::exception&) { rejected=true; }
    require(rejected,"invalid stabilization parameter/ROI accepted");
}
cv::Mat texture() {
    cv::Mat image(240,320,CV_8UC1); cv::RNG rng(3588); rng.fill(image,cv::RNG::UNIFORM,20,235);
    cv::GaussianBlur(image,image,{3,3},0.6); return image;
}
cv::Mat shifted(const cv::Mat& image,double x,double y) {
    cv::Mat out; cv::warpAffine(image,out,cv::Mat(cv::Matx23d(1,0,x,0,1,y)),image.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT_101);
    return out;
}
cv::Point2d shake(int i) {
    constexpr double pi=3.141592653589793;
    return {12*std::sin(2*pi*i/5)+6*std::sin(2*pi*i/3),8*std::cos(2*pi*i/6)+4*std::sin(2*pi*i/4)};
}
double rms(const std::vector<double>& values) {
    const double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
    double sum=0; for (double x:values) sum+=(x-mean)*(x-mean);
    return std::sqrt(sum/values.size());
}
void validRoi(const cv::Rect& r) {
    require(r.width==720 && r.height==540 && r.x>=0 && r.x<=80 && r.y>=0 && r.y<=60 &&
            ((r.x|r.y|r.width|r.height)&1)==0,"ROI dimensions/bounds/alignment");
}
void core() {
    rejects([] { GrayStabilizer bad(0); }); rejects([] { GrayStabilizer bad(1.01); });
    rejects([] { GrayStabilizer bad(NAN); });
    const auto base=texture();
    GrayStabilizer static_camera;
    require(static_camera.update(base,0).roi==cv::Rect(40,30,720,540),"first ROI must be centered");
    for (int i=1;i<6;++i) {
        const auto s=static_camera.update(base,i*1000.0/30);
        require(s.valid && s.roi==cv::Rect(40,30,720,540) && cv::norm(s.delta)<0.1,"static camera drifts");
        require(s.sampled_points<=192,"corner pool exceeded its cap");
    }
    rejects([&] { static_camera.update(cv::Mat(180,320,CV_8UC1),300); });
    for (double sign : {-1.0,1.0}) {
        GrayStabilizer motion; motion.update(base,0);
        const auto s=motion.update(shifted(base,sign*3.2,sign*2.4),1000.0/30);
        require(s.valid && std::abs(s.delta.x-sign*8)<0.8 && std::abs(s.delta.y-sign*6)<0.8,"translation sign or 2.5x scale");
        require(std::abs(s.compensation.x-sign*6.8)<0.8,"EMA translation compensation"); validRoi(s.roi);
    }
    std::vector<double> raw_x,raw_y,out_x,out_y;
    GrayStabilizer jitter;
    int valid=0;
    for (int i=0;i<180;++i) {
        const auto p=shake(i);
        const auto s=jitter.update(shifted(base,p.x/2.5,p.y/2.5),i*1000.0/30);
        valid+=s.valid; validRoi(s.roi);
        if (i>=30) {
            raw_x.push_back(p.x/0.9); raw_y.push_back(p.y/0.9);
            out_x.push_back((p.x-(s.roi.x-40))/0.9); out_y.push_back((p.y-(s.roi.y-30))/0.9);
        }
    }
    require(valid>=170 && rms(out_x)<rms(raw_x)*0.5 && rms(out_y)<rms(raw_y)*0.5,"high-frequency jitter reduction below 50%");
    std::cout<<"jitter RMS x="<<rms(raw_x)<<" -> "<<rms(out_x)<<", y="<<rms(raw_y)<<" -> "<<rms(out_y)<<'\n';
    GrayStabilizer pan;
    StabilizationInfo state;
    for (int i=0;i<90;++i) state=pan.update(shifted(base,i*0.8,0),i*1000.0/30);
    require(state.valid && std::abs(state.compensation.x-2*0.85/0.15)<1.5,"constant pan does not converge/follow");
    const double saved=state.compensation.x;
    state=pan.update(shifted(base,89*0.8,0),90*1000.0/30+300);
    require(state.reference_reset && !state.valid && std::abs(state.compensation.x-saved*0.85)<1e-6,"time gap jumps ROI");
    for (int i=1;i<=30;++i) state=pan.update(cv::Mat(240,320,CV_8UC1,cv::Scalar(90)),3300+i*1000.0/30);
    require(!state.valid && state.roi==cv::Rect(40,30,720,540),"lost texture does not smoothly recenter");
    state=pan.update(base,4400); state=pan.update(base,4433.3333);
    require(state.valid && cv::norm(state.delta)<0.1,"lost flow did not recover");
    GrayStabilizer fast;
    bool clamped=false;
    for (int i=0;i<16;++i) {
        state=fast.update(shifted(base,i*4.8,i*4.8),i*1000.0/30);
        validRoi(state.roi); clamped|=state.clamped;
    }
    require(clamped && state.compensation.x<=40 && state.compensation.y<=30,"large pan exceeds crop margin");
    // A large textured foreground moves over a stationary background. Boxes are
    // provided on the previous stabilized canvas, exercising the inverse ROI mask.
    GrayStabilizer foreground;
    cv::Mat object=texture()(cv::Rect(0,0,160,140)).clone();
    cv::Rect previous_box;
    for (int i=0;i<18;++i) {
        cv::Mat gray=base.clone(); const cv::Rect box(45+i*3,45,160,140); object.copyTo(gray(box));
        std::vector<TrackedDetection> tracks;
        if (i) {
            const auto roi=foreground.previousRoi();
            tracks.push_back({1,{0,.9F,float((previous_box.x*2.5-roi.x)/0.9),float((previous_box.y*2.5-roi.y)/0.9),
                float(((previous_box.x+previous_box.width)*2.5-roi.x)/0.9),float(((previous_box.y+previous_box.height)*2.5-roi.y)/0.9)}});
        }
        state=foreground.update(gray,i*1000.0/30,tracks); previous_box=box;
        if (i) require(state.valid && cv::norm(state.delta)<0.6 && cv::norm(state.compensation)<1,"foreground mistaken for camera shake");
    }
    std::cout<<"PASS translation, EMA, pan, grid/mask, clamp/alignment, loss/recovery and timestamp gaps\n";
}
cv::Mat rgb(const TensorInput& tensor) {
    CpuAccess read(std::static_pointer_cast<DmaBuffer>(tensor.dma.allocation),DMA_BUF_SYNC_READ);
    cv::Mat result=cv::Mat(tensor.dma.height,tensor.dma.width,CV_8UC3,tensor.dma.data,tensor.dma.width_stride*3).clone();
    read.finish(); return result;
}
void cropHardware() {
    auto dma=std::make_shared<DmaBuffer>(832U*624U*3/2,"/dev/dma_heap/system-uncached");
    cv::Mat gray(600,800,CV_8UC1);
    {
        CpuAccess write(dma,DMA_BUF_SYNC_WRITE);
        auto* data=static_cast<unsigned char*>(dma->data);
        std::memset(data,0xCD,dma->size);
        std::memset(data+832U*624U,128,832U*624U/2);
        for (int y=0;y<600;++y) for (int x=0;x<800;++x)
            data[y*832+x]=gray.at<unsigned char>(y,x)=static_cast<unsigned char>(20+x/5+y/10);
        write.finish();
    }
    Nv12FrameView image{dma,dma,800,600,832,624,true};
    auto config=loadAppConfig(std::string(YOLO_SOURCE_DIR)+"/config/camera_coco640.yaml");
    ImagePreprocessor pre(config,{656,642,656U*642U*3});
    yolo::live::Renderer renderer;
    for (cv::Rect roi : {cv::Rect(0,0,720,540),cv::Rect(40,30,720,540),cv::Rect(80,60,720,540)}) {
        auto tensor=pre.prepareNv12(image,roi); const auto pixels=rgb(tensor);
        cv::Mat expected,expected_rgb;
        cv::resize(gray(roi),expected,{640,480}); cv::cvtColor(expected,expected_rgb,cv::COLOR_GRAY2RGB);
        require(cv::norm(pixels.rowRange(80,560),expected_rgb,cv::NORM_INF)<=3,"model crop/resize/color/stride mismatch");
        require(cv::norm(pixels.rowRange(0,80),cv::Mat(80,640,CV_8UC3,cv::Scalar::all(114)),cv::NORM_INF)==0,"crop damaged letterbox");
        yolo::live::DecodedFrame frame; frame.image=image; frame.stabilization.roi=roi;
        const auto output=renderer.convert(frame);
        cv::resize(gray(roi),expected,{800,600}); cv::cvtColor(expected,expected_rgb,cv::COLOR_GRAY2BGR);
        require(cv::norm(output,expected_rgb,cv::NORM_INF)<=3,"renderer and detector crop differ"); renderer.finish();
        const float boxes[]{80,160,240,400}; std::vector<float> scores(80,0); scores[0]=.9F;
        const auto found=decodeYolo26(boxes,scores.data(),1,tensor,config);
        require(found.size()==1 && std::abs(found[0].x1-100)<.001 && std::abs(found[0].y1-100)<.001,
                "letterbox no longer returns stabilized 800x600 canvas coordinates");
    }
    for (cv::Rect bad : {cv::Rect(1,0,720,540),cv::Rect(0,1,720,540),cv::Rect(82,60,720,540),cv::Rect(0,0,721,540)})
        rejects([&] { pre.prepareNv12(image,bad); });
    FrameStabilizer small;
    for (int i=0;i<8;++i) { const auto s=small.update(image,i*1000.0/30); validRoi(s.roi); }
    small.finish();
    std::cout<<"PASS actual RGA NV12 crops, padded planes, RGB canvas, letterbox and two-slot gray DMA\n";
}
void artifacts(const std::filesystem::path& directory,int count=180) {
    using namespace yolo::live;
    require(std::filesystem::create_directory(directory),"artifact directory must be new");
    const auto fixture=directory/"fixture"; std::filesystem::create_directory(fixture);
    cv::Mat base=cv::imread("/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg");
    require(!base.empty(),"missing installed bus fixture");
    // Remove the fixture's baked-in black side bars before generating shake.
    cv::Mat luminance; cv::cvtColor(base,luminance,cv::COLOR_BGR2GRAY);
    base=base(cv::boundingRect(luminance)).clone(); cv::resize(base,base,{800,600});
    std::ofstream manifest(fixture/"manifest.jsonl");
    manifest<<"{\"type\":\"mjpeg_fixture\",\"width\":800,\"height\":600,\"negotiated_fps\":30,\"max_packet_bytes\":960000}\n";
    for (int i=0;i<count;++i) {
        const auto p=shake(i); const auto image=shifted(base,p.x,p.y);
        require(cv::imwrite((fixture/(std::to_string(i)+".jpg")).string(),image,{cv::IMWRITE_JPEG_QUALITY,95}),"JPEG fixture write");
        manifest<<nlohmann::json{{"file",std::to_string(i)+".jpg"},{"sequence",i},{"timestamp_ms",i*1000.0/30}}.dump()<<'\n';
    }
    manifest.close();
    auto source=replaySource(fixture/"manifest.jsonl",false); MjpegDecoder decoder; FrameStabilizer stabilizer;
    Renderer fixed,stable;
    cv::VideoWriter video((directory/"comparison.avi").string(),cv::VideoWriter::fourcc('M','J','P','G'),30,{1600,600});
    require(video.isOpened(),"comparison video open failed");
    std::ofstream diagnostics(directory/"roi.jsonl");
    std::atomic<bool> stop{false};
    std::vector<double> original_x,original_y,output_x,output_y;
    for (int i=0;i<count;++i) {
        auto packet=source->next(stop); require(packet.has_value(),"missing JPEG packet");
        auto frame=decoder.decode(*packet); require(frame.has_value(),"decode surface exhausted");
        const auto s=stabilizer.update(frame->image,i*1000.0/30); validRoi(s.roi);
        if (i==30) {
            const auto config=loadAppConfig(std::string(YOLO_SOURCE_DIR)+"/config/camera_coco640.yaml");
            YoloEngine serial(config);
            ImagePreprocessor pre(config,serial.inputMemoryLayout());
            auto tensor=pre.prepareNv12(frame->image,s.roi);
            const auto expected=serial.infer(tensor,i);
            require(!expected.detections.empty(),"cropped bus produced no COCO detections");
            for (int core:{1,2,4}) {
                YoloEngine split(config,core); ImagePreprocessor input(config,split.inputMemoryLayout());
                auto pixels=input.prepareNv12(frame->image,s.roi);
                const auto actual=split.infer(pixels,i);
                require(actual.detections.size()==expected.detections.size(),"cropped single/split detection count differs");
                for (size_t n=0;n<actual.detections.size();++n) {
                    const auto& a=actual.detections[n]; const auto& b=expected.detections[n];
                    require(a.class_id==b.class_id && std::abs(a.score-b.score)<=.01 &&
                        std::abs(a.x1-b.x1)<=1 && std::abs(a.y1-b.y1)<=1 &&
                        std::abs(a.x2-b.x2)<=1 && std::abs(a.y2-b.y2)<=1,
                        "cropped single/split class/score/coordinate mismatch");
                }
            }
        }
        frame->stabilization.roi={40,30,720,540}; const auto before=fixed.convert(*frame);
        frame->stabilization=s; const auto after=stable.convert(*frame);
        cv::Mat joined; cv::hconcat(before,after,joined);
        fixed.finish(); stable.finish();
        cv::putText(joined,"Fixed center crop",{15,30},cv::FONT_HERSHEY_SIMPLEX,.7,{0,255,0},2);
        cv::putText(joined,"CPU LK + EMA / RGA ROI",{815,30},cv::FONT_HERSHEY_SIMPLEX,.7,{0,255,0},2);
        video.write(joined);
        if (i==44) require(cv::imwrite((directory/"comparison.jpg").string(),joined),"comparison still write");
        const auto p=shake(i);
        if (i>=30) {
            original_x.push_back(p.x/0.9); original_y.push_back(p.y/0.9);
            output_x.push_back((p.x-(s.roi.x-40))/0.9); output_y.push_back((p.y-(s.roi.y-30))/0.9);
        }
        diagnostics<<nlohmann::json{{"sequence",i},{"motion_valid",s.valid},{"roi",{s.roi.x,s.roi.y,720,540}},
            {"known_shift",{p.x,p.y}},{"estimated_delta",{s.delta.x,s.delta.y}},{"total_ms",s.total_ms}}.dump()<<'\n';
    }
    video.release();
    stabilizer.finish();
    const bool passed=rms(output_x)<rms(original_x)*.5 && rms(output_y)<rms(original_y)*.5;
    std::ofstream(directory/"report.json")<<nlohmann::json{{"passed",passed},{"frames",count},
        {"fixed_crop_rms",{rms(original_x),rms(original_y)}},{"stabilized_rms",{rms(output_x),rms(output_y)}}}.dump(2)<<'\n';
    require(passed,"MPP/RGA real-image stabilization RMS did not decrease 50%");
    std::cout<<"PASS MPP/RGA stabilization comparison: "<<directory<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        cv::setNumThreads(2);
        if (argc==1) core();
        else if (argc==2 && std::string(argv[1])=="--hardware") cropHardware();
        else if (argc==3 && std::string(argv[1])=="--artifacts") artifacts(argv[2]);
        else throw std::runtime_error("Usage: stabilization_test [--hardware | --artifacts NEW_DIR]");
        return 0;
    } catch (const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
