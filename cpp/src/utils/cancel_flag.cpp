#include "cancel_flag.hpp"

#include <atomic>
#include <csignal>

namespace vhs {
namespace utils {

namespace {
std::atomic<bool> g_cancel_requested{false};

void handle_signal(int) {
    g_cancel_requested.store(true, std::memory_order_relaxed);
}
} // namespace

void install_signal_handlers() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
}

bool is_cancel_requested() {
    return g_cancel_requested.load(std::memory_order_relaxed);
}

void request_cancel() {
    g_cancel_requested.store(true, std::memory_order_relaxed);
}

} // namespace utils
} // namespace vhs
