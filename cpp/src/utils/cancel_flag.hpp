#pragma once

namespace vhs {
namespace utils {

void install_signal_handlers();
bool is_cancel_requested();
void request_cancel();

} // namespace utils
} // namespace vhs
