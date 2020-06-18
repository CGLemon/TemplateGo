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
bool is_allnumber(std::string &);
bool is_number(char); 
bool is_float(std::string &);

float cached_t_quantile(int v);


template <class T> void atomic_add(std::atomic<T> &f, T d) {
  T old = f.load();
  while (!f.compare_exchange_weak(old, old + d))
    ;
}


class Exception : public std::runtime_error {
public:
  Exception(const std::string &what);
};


}

#endif
