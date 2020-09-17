#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"

#include <random>
#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>
#include <atomic>
#include <stdexcept>
#include <string>

namespace Utils {

void static_printf(const char *fmt, ...);

void auto_printf(const char *fmt, ...);

void stream_printf(const char *fmt, ...);

void gtp_printf(const char *fmt, ...);

void gtp_fail_printf(const char *fmt, ...);

bool is_integer(std::string &number_str);

bool is_unsigned_integer(std::string &number_str);

bool is_number(char);

bool is_float(std::string &float_str);

float cached_t_quantile(int v);

std::uint64_t rng_seed(int times = 1);


template <class T> 
void atomic_add(std::atomic<T> &f, T d) {
  T old = f.load();
  while (!f.compare_exchange_weak(old, old + d))
    ;
}


class Timer {
public:
  Timer();

  void clock();  
   
  int get_duration_seconds() const;
  
  int get_duration_milliseconds() const;

  int get_duration_microseconds () const;

  float get_duration() const;

  void record();

  void release();

  float get_record_time(int) const;
  
  int get_record_count() const;
  
  float* get_record();

private:
  std::chrono::steady_clock::time_point m_clock_time;

  std::vector<float> m_record;

  size_t record_count;
};


}

#endif
