#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json_fwd.hpp>
namespace yolo::trace {
enum Kind : uint16_t {
    Instant,
    Begin,
    End,
    Enter,
    Leave,
    Evict,
    Acquire,
    Release,
    Drop,
    Error,
    Counter
};
enum Stage : uint16_t {
    Thread,
    Poll,
    Dqbuf,
    Capture,
    SourceTime,
    SourceFlags,
    Valid,
    Decode,
    Submit,
    GetFrame,
    Stabilize,
    Rga,
    Input,
    Run,
    OutputGet,
    Postprocess,
    Detection,
    Reorder,
    Track,
    Snapshot,
    Render,
    Output,
    Encode,
    Encoded,
    PacketQueue,
    InferQueue,
    OutputQueue,
    ResultQueue,
    EncodeQueue,
    PacketPool,
    SurfacePool,
    RgbPool,
    EncodePool,
    BufferSearch,
    Lock,
    WorkerTask,
    WorkerWait,
    Write,
    QueueDepth,
    PoolCapacity,
    SourceGap,
    CaptureSlot,
    Correction,
    Prediction,
    SourceRawSequence,
    WorkerMask,
    Stop,
    FrameError,
    Gaps,
    Rejected,
    StreamSubmit
};
enum Lane : uint32_t {
    Main = 0,
    Camera = 1,
    Decoder = 2,
    Worker0 = 3,
    Worker1 = 4,
    Worker2 = 5,
    Tracker = 6,
    Writer = 7,
    Encoder = 8,
    Lanes = 9
};
struct Record {
    uint64_t timestamp_ns, frame_id, span, value;
    uint32_t thread_id, resource_id, serial;
    uint16_t kind, stage;
};
static_assert(sizeof(Record) == 48 && alignof(Record) == 8 && offsetof(Record, kind) == 44);
class Session {
  public:
    Session(const std::filesystem::path &directory, size_t capacity, unsigned sample_ms = 250);
    ~Session();
    void finish(const nlohmann::json &summary);
    void metadata(const nlohmann::json &data);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#if YOLO_ENABLE_TRACE
extern std::atomic<bool> recording;
inline bool enabled() { return recording.load(std::memory_order_relaxed); }
void thread(Lane lane);
uint64_t emit(Kind, Stage, uint64_t frame = 0, uint64_t span = 0, uint64_t value = 0,
              uint32_t resource = 0) noexcept;
uint64_t currentFrame() noexcept;
void frame(uint64_t id) noexcept;
uint32_t lane() noexcept;
#else
inline constexpr bool enabled() { return false; }
inline void thread(Lane) {}
inline uint64_t emit(Kind, Stage, uint64_t = 0, uint64_t = 0, uint64_t = 0, uint32_t = 0) { return 0; }
inline uint64_t currentFrame() { return 0; }
inline void frame(uint64_t) {}
inline uint32_t lane() { return Main; }
#endif
class Scope {
  public:
    explicit Scope(Stage stage, uint64_t id = currentFrame(), uint32_t resource = 0)
        : stage_(stage), id_(id), resource_(resource),
          span_(enabled() ? emit(Begin, stage, id, 0, 0, resource) : 0) {}
    void end() {
        if (span_) {
            emit(End, stage_, id_, span_, 0, resource_);
            span_ = 0;
        }
    }
    ~Scope() { end(); }
    Scope(const Scope &) = delete;

  private:
    Stage stage_;
    uint64_t id_;
    uint32_t resource_;
    uint64_t span_;
};
// 租约的控制块来自固定内存池。最后一个消费者归还租约时先记录 Release，
// 随后才释放底层 shared_ptr，保证同一槽的下一次 Acquire 可以与本次配对。
std::shared_ptr<void> leaseOwner(std::shared_ptr<void>, Stage, uint32_t slot, uint64_t id);
template <class T>
std::shared_ptr<T> lease(const std::shared_ptr<T> &p, Stage stage, uint32_t slot, uint64_t id) {
    if (!enabled())
        return p;
    auto owner = leaseOwner(p, stage, slot, id);
    return std::shared_ptr<T>(owner, p.get());
}
} // namespace yolo::trace
