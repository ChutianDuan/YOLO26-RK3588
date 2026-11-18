#include "camera/stream_output.h"
#include "image/dma_buffer.h"
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace yolo;
using namespace yolo::live;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected artifact directory");
        const std::filesystem::path dir(argv[1]); std::filesystem::create_directories(dir);
        EncodeFramePool pool;
        auto a = pool.acquire(), b = pool.acquire(), c = pool.acquire();
        require(a && b && c && !pool.acquire(), "pool overwrote a leased slot");
        auto registered_allocation = a->dma;
        auto copied_frame = a;
        a.reset();
        require(!pool.acquire(), "encoder pool reused slot while a frame copy remained");
        copied_frame.reset();
        a = pool.acquire();
        require(a && a->dma == registered_allocation, "allocation reference incorrectly blocks frame reuse");
        a.reset(); b.reset(); c.reset();

        cv::Mat bars(600, 800, CV_8UC3);
        const std::array<cv::Scalar,5> colors{cv::Scalar(0,0,0), cv::Scalar(255,255,255),
            cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(255,0,0)};
        for (int i=0; i<5; ++i) bars(cv::Rect(i*160,0,160,600)).setTo(colors[i]);
        cv::Mat yuv; cv::cvtColor(bars, yuv, cv::COLOR_BGR2YUV_I420);
        auto dma = std::make_shared<DmaBuffer>(832U*624U*3/2, "/dev/dma_heap/system-uncached");
        {
            CpuAccess access(dma, DMA_BUF_SYNC_WRITE);
            auto* raw = static_cast<uint8_t*>(dma->data);
            std::memset(raw, 16, 832*624); std::memset(raw+832*624,128,832*624/2);
            const auto* planar = yuv.ptr<uint8_t>();
            for (int y=0; y<600; ++y) std::memcpy(raw+y*832,planar+y*800,800);
            for (int y=0; y<300; ++y) for (int x=0; x<400; ++x) {
                raw[832*624+y*832+x*2] = planar[800*600+y*400+x];
                raw[832*624+y*832+x*2+1] = planar[800*600+800*600/4+y*400+x];
            }
            access.finish();
        }
        DecodedFrame source; source.image={dma,dma,800,600,832,624,false};
        Renderer renderer;
        auto pixels = renderer.convert(source);
        cv::rectangle(pixels, {340,160}, {450,290}, {40,220,40}, 2);
        cv::putText(pixels,"17 person",{340,150},cv::FONT_HERSHEY_SIMPLEX,.5,{40,220,40},1);
        cv::imwrite((dir/"reference.png").string(), pixels);
        auto encoded_frame = pool.acquire();
        bool rejected=false;
        try { renderer.convertForEncode(*encoded_frame); } catch (const std::exception&) { rejected=true; }
        require(rejected,"encoding while CPU RW active accepted");
        renderer.finish(); renderer.convertForEncode(*encoded_frame);

        const int fd=socket(AF_INET,SOCK_DGRAM|SOCK_CLOEXEC,0); require(fd>=0,"receiver socket");
        sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        require(bind(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0,"receiver bind");
        socklen_t size=sizeof(addr); getsockname(fd,reinterpret_cast<sockaddr*>(&addr),&size);
        int rcvbuf=1024*1024; setsockopt(fd,SOL_SOCKET,SO_RCVBUF,&rcvbuf,sizeof(rcvbuf));
        std::atomic<bool> done{false}, valid{true};
        std::thread receiver([&] {
            std::ofstream ts(dir/"encoded.ts",std::ios::binary);
            std::array<char,2048> data{};
            while (true) {
                pollfd p{fd,POLLIN,0};
                if (poll(&p,1,50)<=0) { if (done.load()) break; continue; }
                const auto n=recv(fd,data.data(),data.size(),0);
                if (n<=0 || n>1316 || n%188) { valid=false; continue; }
                ts.write(data.data(),n);
            }
        });
        nlohmann::json expected=nlohmann::json::array();
        try {
            MppH264Encoder encoder;
            MpegTsOutput output(ntohs(addr.sin_port));
            std::ofstream h264(dir/"encoded.h264", std::ios::binary);
            for (int i=0;i<64;++i) {
                // Deliberate 100-ms gap after frame 1; timestamps must not become CFR.
                const int64_t ticks=i*3000LL+(i>=2?6000:0);
                StreamFrame f{*encoded_frame,{i,1000+ticks/90.0,0,0,"synthetic"}};
                const auto& packet=encoder.encode(f);
                require(packet.pts90==ticks,"encoder changed source timestamp");
                h264.write(reinterpret_cast<const char*>(packet.bytes.data()),packet.bytes.size());
                require(output.write(packet),"UDP send failed");
                expected.push_back({{"pts90",packet.pts90},{"keyframe",packet.keyframe}});
            }
        } catch (...) { done=true; receiver.join(); close(fd); throw; }
        done=true; receiver.join(); close(fd);
        require(valid.load(),"invalid UDP MPEG-TS packet size/alignment");
        std::ofstream(dir/"expected.json")<<expected.dump(2);
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
