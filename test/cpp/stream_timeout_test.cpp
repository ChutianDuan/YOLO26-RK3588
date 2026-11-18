#include "camera/stream_output.h"
#include "image/dma_buffer.h"
#include <iostream>
#include <rk_mpi.h>
using namespace yolo::live;
namespace {
MppApi api_copy{};
MppApi* real_api=nullptr;
MppCtx recorded=nullptr;
int resets=0, destroys=0;
bool fail_reset=false;
MPP_RET timeout(MppCtx, MppPacket*) { return MPP_ERR_TIMEOUT; }
MPP_RET reset(MppCtx ctx) { ++resets; return fail_reset ? MPP_NOK : real_api->reset(ctx); }
void require(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
}
extern "C" MPP_RET __real_mpp_create(MppCtx*,MppApi**);
extern "C" MPP_RET __real_mpp_destroy(MppCtx);
extern "C" MPP_RET __wrap_mpp_create(MppCtx* ctx,MppApi** api) {
    auto rc=__real_mpp_create(ctx,api);
    if(rc==MPP_OK) {
        recorded=*ctx; real_api=*api; api_copy=**api;
        api_copy.encode_get_packet=timeout; api_copy.reset=reset; *api=&api_copy;
    }
    return rc;
}
extern "C" MPP_RET __wrap_mpp_destroy(MppCtx ctx) { ++destroys; return __real_mpp_destroy(ctx); }
int main() {
    try {
        for(bool quarantine:{false,true}) {
            EncodeFramePool pool;
            auto encoder=std::make_unique<MppH264Encoder>();
            auto image=pool.acquire();
            bool threw=false;
            try { encoder->encode(StreamFrame{*image,{1,1000,0,0,"test"}}); }
            catch(const std::exception&) { threw=true; }
            require(threw,"timeout not propagated"); image.reset();
            auto a=pool.acquire(), b=pool.acquire();
            require(a&&b&&!pool.acquire(),"in-flight input reused before encoder reset");
            const int old_reset=resets,old_destroy=destroys;
            fail_reset=quarantine;
            encoder.reset();
            require(resets==old_reset+1,"reset not attempted");
            if(quarantine) {
                require(destroys==old_destroy&&!pool.acquire(),"uncertain DMA slot not quarantined");
                // Test-only cleanup of the real device. The intentionally retained
                // Impl/leases remain quarantined until this short-lived test exits.
                fail_reset=false; require(real_api->reset(recorded)==MPP_OK,"real reset failed");
                require(__real_mpp_destroy(recorded)==MPP_OK,"real destroy failed");
            } else {
                require(destroys==old_destroy+1&&pool.acquire().has_value(),"confirmed cleanup did not release input");
            }
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
