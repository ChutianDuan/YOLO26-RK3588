#include "video/fair_scheduler.h"
#include <array>
#include <atomic>
#include <iostream>
#include <memory>
#include <set>
#include <thread>

using yolo::multi::FairScheduler;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        FairScheduler<int> fair(4);
        for (int i = 0; i < 4; ++i) fair.put(i, i);
        for (int n = 0; n < 4000; ++n) {
            auto task = fair.take(); require(task && task->channel == size_t(n % 4), "round robin starvation");
            fair.put(task->channel, n); fair.put(task->channel, n + 1);
            fair.complete(task->channel);
        }
        auto counts = fair.stats();
        for (const auto& s : counts) require(s.dispatched == 1000 && s.replaced == 1000, "dispatch/replacement accounting");
        fair.cancel(); require(!fair.take(), "cancel did not wake consumers");

        FairScheduler<int> draining(2);
        draining.put(0, 7); draining.finish(0); draining.finish(1);
        auto final = draining.take(); require(final && final->value == 7, "EOF lost last pending frame");
        require(!draining.finished(0), "inflight frame prematurely finished");
        draining.complete(0); require(!draining.take(), "EOF not drained");

        FairScheduler<std::shared_ptr<int>> leases(1);
        auto pixels = std::make_shared<int>(42);
        leases.put(0, pixels); auto active = leases.take();
        leases.put(0, std::make_shared<int>(13)); leases.cancel();
        require(active && *active->value == 42 && pixels.use_count() == 2, "cancel invalidated active lease");
        require(leases.stats()[0].busy && leases.stats()[0].cancelled == 1, "cancel accounting");
        active.reset(); leases.complete(0); require(pixels.use_count() == 1, "active lease leaked");

        FairScheduler<int> parallel(4);
        std::array<std::atomic<int>,4> busy{};
        std::atomic<int> errors{0};
        std::mutex seen_mutex;
        std::set<std::pair<size_t,int>> seen;
        // Make all channels ready before starting workers, then replace tasks concurrently.
        for (int i = 0; i < 4; ++i) parallel.put(i, 0);
        std::array<std::thread,3> workers;
        for (auto& worker : workers) worker = std::thread([&] {
            while (auto task = parallel.take()) {
                if (busy[task->channel].fetch_add(1) != 0) ++errors;
                { std::lock_guard<std::mutex> lock(seen_mutex);
                  if (!seen.emplace(task->channel, task->value).second) ++errors; }
                for (int i = 0; i < int(task->channel + 1); ++i) std::this_thread::yield();
                --busy[task->channel]; parallel.complete(task->channel);
            }
        });
        std::array<std::thread,4> producers;
        for (size_t i = 0; i < producers.size(); ++i) producers[i] = std::thread([&, i] {
            for (int n = 1; n <= 4000; ++n) { parallel.put(i, n); if (n % 17 == 0) std::this_thread::yield(); }
            parallel.finish(i);
        });
        for (auto& p : producers) p.join();
        for (auto& w : workers) w.join();
        require(errors == 0, "duplicate dispatch or concurrent inference on a channel");
        for (const auto& s : parallel.stats())
            require(s.eof && !s.pending && !s.busy && s.dispatched > 0 && s.dispatched + s.replaced == 4001,
                    "concurrent drain/accounting failure");
        std::cout << "PASS fair cursor, EOF, bounded replacement, cancellation leases and three concurrent workers\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
