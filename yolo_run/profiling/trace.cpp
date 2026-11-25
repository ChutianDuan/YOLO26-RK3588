#include "profiling/trace.h"
#include "profiling/sampler.h"
#include <array>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
namespace yolo::trace {
using Json = nlohmann::json;
#if YOLO_ENABLE_TRACE
namespace {
constexpr const char *names[] = {"Thread",
                                 "Poll",
                                 "Dqbuf",
                                 "Capture",
                                 "SourceTime",
                                 "SourceFlags",
                                 "Valid",
                                 "Decode",
                                 "MPP submit",
                                 "MPP get frame",
                                 "Stabilize",
                                 "RGA",
                                 "NPU input",
                                 "NPU run",
                                 "NPU output",
                                 "Postprocess",
                                 "Detection",
                                 "Reorder",
                                 "Track",
                                 "Snapshot",
                                 "Render",
                                 "Output",
                                 "Encode",
                                 "Encoded",
                                 "Packet queue",
                                 "Inference queue",
                                 "Output queue",
                                 "Result queue",
                                 "Encode queue",
                                 "Packet pool",
                                 "Surface pool",
                                 "RGB pool",
                                 "Encode pool",
                                 "Buffer search",
                                 "Lock",
                                 "Worker task",
                                 "Worker wait",
                                 "JSONL write",
                                 "Queue depth",
                                 "Pool capacity",
                                 "Source gap",
                                 "Capture slot",
                                 "Correction",
                                 "Prediction",
                                 "Raw sequence",
                                 "Core mask",
                                 "Stop",
                                 "Frame error",
                                 "Trace events dropped",
                                 "Rejected capture",
                                 "Stream submit"};
static_assert(std::size(names) == StreamSubmit + 1);
constexpr const char *lanes[] = {"Main / output", "V4L2 or replay", "MPP / stabilization",
                                 "NPU worker 0",  "NPU worker 1",   "NPU worker 2",
                                 "ByteTrack",     "JSONL writer",   "MPP encoder"};
uint64_t now() noexcept {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct Ring {
    std::unique_ptr<Record[]> data;
    size_t capacity = 0;
    alignas(64) std::atomic<uint64_t> write{0};
    uint64_t lost = 0, last_loss_ns = 0;
    uint32_t serial = 0;
    alignas(64) std::atomic<uint64_t> read{0};
};
Ring *rings = nullptr;
thread_local Lane myLane = Main;
thread_local uint32_t tid = 0;
thread_local uint64_t current = 0;
struct Block {
    alignas(std::max_align_t) unsigned char data[128];
    std::atomic<bool> used{false};
};
std::array<Block, 256> blocks;
template <class T> struct Alloc {
    using value_type = T;
    Alloc() = default;
    template <class U> Alloc(const Alloc<U> &) {}
    T *allocate(size_t n) {
        if (sizeof(T) * n > sizeof(Block::data))
            throw std::bad_alloc();
        for (auto &b : blocks) {
            bool free = false;
            if (b.used.compare_exchange_strong(free, true))
                return reinterpret_cast<T *>(b.data);
        }
        throw std::bad_alloc();
    }
    void deallocate(T *ptr, size_t) { reinterpret_cast<Block *>(ptr)->used.store(false); }
};
struct Holder {
    std::shared_ptr<void> owner;
    Stage pool;
    uint32_t slot;
    uint64_t frame, generation;
    Holder(std::shared_ptr<void> p, Stage s, uint32_t r, uint64_t f)
        : owner(std::move(p)), pool(s), slot(r), frame(f), generation(emit(Acquire, s, f, 0, 0, r)) {}
    ~Holder() { emit(Release, pool, frame, generation, 0, slot); }
};
void jsonFile(const std::filesystem::path &p, const Json &data) {
    auto temporary = p;
    temporary += ".tmp";
    std::ofstream out(temporary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << data.dump(2) << '\n';
    out.close();
    // 进程被中断时仍保留上一份完整的事件字典，便于离线解析已写出的记录。
    std::filesystem::rename(temporary, p);
}
} // namespace
std::atomic<bool> recording{false};
uint32_t lane() noexcept { return myLane; }
void frame(uint64_t id) noexcept { current = id; }
uint64_t currentFrame() noexcept { return current; }
void thread(Lane l) {
    myLane = l;
    current = 0;
    tid = static_cast<uint32_t>(syscall(SYS_gettid));
    emit(Instant, Thread, 0, 0, l);
}
uint64_t emit(Kind k, Stage s, uint64_t f, uint64_t span, uint64_t value, uint32_t resource) noexcept {
    if (!enabled())
        return 0;
    auto &ring = rings[myLane];
    const auto serial = ++ring.serial;
    if (!span)
        span = (uint64_t(myLane + 1) << 48) | serial;
    // 每条轨道只有一个生产线程。先填完记录再发布写指针，收集线程据此读取完整记录。
    auto w = ring.write.load(std::memory_order_relaxed);
    if (w - ring.read.load(std::memory_order_acquire) >= ring.capacity) {
        ++ring.lost;
        ring.last_loss_ns = now();
        return span;
    }
    ring.data[w % ring.capacity] = {now(), f, span, value, tid, resource, serial, k, s};
    ring.write.store(w + 1, std::memory_order_release);
    return span;
}
std::shared_ptr<void> leaseOwner(std::shared_ptr<void> p, Stage s, uint32_t slot, uint64_t id) {
    return std::allocate_shared<Holder>(Alloc<Holder>{}, std::move(p), s, slot, id);
}
struct Session::Impl {
    std::filesystem::path dir;
    std::array<Ring, Lanes> buffers;
    std::ofstream file;
    std::thread collector;
    std::atomic<bool> stop{false};
    Json config;
    std::atomic<bool> sample_error{false};
    bool done = false;
    Impl(const std::filesystem::path &path, size_t capacity, unsigned sample_ms) : dir(path) {
        if (sample_ms < 20 || sample_ms > 60000 || recording || capacity < 2 || capacity > 262144)
            throw std::runtime_error("invalid trace capacity or active session");
        if (!std::filesystem::create_directory(dir))
            throw std::runtime_error("trace directory must be new");
        file.open(dir / "trace.bin", std::ios::binary);
        file.write("Y26TRACE", 8);
        const uint32_t header[] = {1, 48, 0x01020304, CLOCK_MONOTONIC};
        file.write(reinterpret_cast<const char *>(header), sizeof(header));
        if (!file)
            throw std::runtime_error("cannot create trace.bin");
        for (auto &r : buffers) {
            r.data = std::make_unique<Record[]>(capacity);
            r.capacity = capacity;
        }
        config = {{"version", 1},
                  {"clock", "CLOCK_MONOTONIC"},
                  {"ring_capacity", capacity},
                  {"sample_ms", sample_ms},
                  {"stages", names},
                  {"lanes", lanes},
                  {"status", "incomplete"}};
        jsonFile(dir / "config.json", config);
        collector = std::thread([this, sample_ms] {
            std::unique_ptr<Sampler> sampler;
            try {
                sampler = std::make_unique<Sampler>(dir);
            } catch (...) {
                sample_error = true;
            }
            auto next = std::chrono::steady_clock::now();
            for (;;) {
                const bool last = stop.load(std::memory_order_acquire);
                for (auto &r : buffers) {
                    auto pos = r.read.load(std::memory_order_relaxed);
                    const auto limit = r.write.load(std::memory_order_acquire);
                    while (pos < limit) {
                        const auto n = std::min(limit - pos, r.capacity - pos % r.capacity);
                        file.write(reinterpret_cast<const char *>(&r.data[pos % r.capacity]),
                                   n * sizeof(Record));
                        pos += n;
                    }
                    r.read.store(pos, std::memory_order_release);
                }
                auto time = std::chrono::steady_clock::now();
                if (sampler && time >= next) {
                    try {
                        sampler->sample();
                        if (!sampler->good())
                            sample_error = true;
                    } catch (...) {
                        sample_error = true;
                        sampler.reset();
                    }
                    next = time + std::chrono::milliseconds(sample_ms);
                }
                if (last)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
        rings = buffers.data();
        recording = true;
        thread(Main);
    }
    void close() {
        if (!done) {
            recording = false;
            stop = true;
            collector.join();
            file.close();
            rings = nullptr;
            done = true;
        }
    }
    ~Impl() { close(); }
};
Session::Session(const std::filesystem::path &p, size_t c, unsigned period)
    : impl_(std::make_unique<Impl>(p, c, period)) {}
void Session::metadata(const Json &j) {
    impl_->config.update(j);
    jsonFile(impl_->dir / "config.json", impl_->config);
}
void Session::finish(const Json &summary) {
    auto &i = *impl_;
    i.close();
    if (!i.file)
        throw std::runtime_error("trace write failed");
    Json counts = Json::array();
    for (auto &r : i.buffers)
        counts.push_back({{"attempted", r.serial}, {"lost", r.lost}, {"last_loss_ns", r.last_loss_ns}});
    i.config.update({{"status", summary.value("status", "error")},
                     {"sample_error", i.sample_error.load()},
                     {"producers", counts},
                     {"pipeline", summary}});
    jsonFile(i.dir / "config.json", i.config);
}
#else
struct Session::Impl {};
Session::Session(const std::filesystem::path &, size_t, unsigned) {
    throw std::runtime_error("compiled without YOLO_ENABLE_TRACE");
}
void Session::metadata(const Json &) {}
void Session::finish(const Json &) {}
std::shared_ptr<void> leaseOwner(std::shared_ptr<void> p, Stage, uint32_t, uint64_t) { return p; }
#endif
Session::~Session() = default;
} // namespace yolo::trace
