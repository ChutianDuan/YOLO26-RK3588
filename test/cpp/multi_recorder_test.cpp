#include "video/mjpeg_recorder.h"
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <opencv2/videoio.hpp>
int main() {
    const auto directory = std::filesystem::temp_directory_path() / ("multi_recorder_" + std::to_string(getpid()));
    try {
        cv::setNumThreads(2); std::filesystem::create_directory(directory);
        cv::Mat image(240,320,CV_8UC3,cv::Scalar(20,120,210));
        const auto path = directory / "fractional.avi";
        {
            yolo::multi::MjpegRecorder recorder(path,30000.0/1001,image.size());
            for (int i=0;i<17;++i) recorder.write(image,i);
            bool rejected = false;
            try { recorder.write(image,16); } catch (const std::exception&) { rejected=true; }
            if (!rejected) throw std::runtime_error("duplicate output sequence accepted");
            recorder.finish(); recorder.finish();
        }
        cv::VideoCapture capture(path.string(),cv::CAP_FFMPEG,{cv::CAP_PROP_N_THREADS,1});
        if (!capture.isOpened() || std::abs(capture.get(cv::CAP_PROP_FPS)-30000.0/1001)>.001)
            throw std::runtime_error("fractional AVI FPS changed");
        cv::Mat decoded; int count=0;
        while (capture.read(decoded)) {
            if (decoded.size()!=image.size() || cv::norm(cv::mean(decoded)-cv::mean(image))>12)
                throw std::runtime_error("recorded dimensions/colors corrupted");
            ++count;
        }
        if (count!=17) throw std::runtime_error("AVI frame count incorrect");
        bool failed=false;
        try {
            yolo::multi::MjpegRecorder full("/dev/full",30,image.size());
            full.write(image,0); full.finish();
        } catch (const std::exception&) { failed=true; }
        if (!failed) throw std::runtime_error("disk write error not propagated");
        std::filesystem::remove_all(directory);
        std::cout<<"PASS MJPG AVI fractional FPS, frame count, colors, sequence and actual ENOSPC\n";return 0;
    } catch(const std::exception& e) {
        std::filesystem::remove_all(directory);std::cerr<<e.what()<<'\n';return 1;
    }
}
