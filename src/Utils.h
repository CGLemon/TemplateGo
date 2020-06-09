#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"
#include <atomic>
#include <stdexcept>
#include <string>

namespace Utils {

void auto_printf(const char *fmt, ...);
void stream_printf(const char *fmt, ...);
void gtp_printf(const char *fmt, ...);
void gtp_fail_printf(const char *fmt, ...);

template <class T> void atomic_add(std::atomic<T> &f, T d) {
  T old = f.load();
  while (!f.compare_exchange_weak(old, old + d))
    ;
}

// Exception to throw around.
class Exception : public std::runtime_error {
public:
  Exception(const std::string &what);
};

} // namespace Utils

#endif
