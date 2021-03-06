#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <atomic>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <chrono>


namespace Utils {

void auto_printf(const char *fmt, ...);

void auto_printf(std::ostringstream &out);

void gtp_output(const char *fmt, ...);

void gtp_fail(const char *fmt, ...);

void space_stream(std::ostream &out, const size_t times);

void strip_stream(std::ostream &out, const size_t times);

float cached_t_quantile(int v);

template <typename T> 
void adjust_range(T &a, const T max, const T min = (T)0) {
    assert(max > min);
    if (a > max) {
        a = max;
    }
    else if (a < min) {
        a = min;
    }
}

template <typename T> 
void atomic_add(std::atomic<T> &f, T d) {
    T old = f.load();
    while (!f.compare_exchange_weak(old, old + d)) {}
}

/**
 * Transform the string to words, and store one by one.
 */
class CommandParser {
public:
    

    struct Reuslt {
        Reuslt(const std::string &s, const int i) : str(s), idx(i) {};

        Reuslt(const std::string &&s, const int i)
            : str(std::forward<decltype(s)>(s)), idx(i) {};

        std::string str;
        int idx;

        template<typename T> T get() const;
    };

    CommandParser() = delete;

    CommandParser(std::string input);

    CommandParser(int argc, char** argv);

    bool valid() const;

    size_t get_count() const;

    std::shared_ptr<Reuslt> get_command(size_t id) const;
    std::shared_ptr<Reuslt> get_commands(size_t begin = 0) const;
    std::shared_ptr<Reuslt> get_slice(size_t begin, size_t end) const;
    std::shared_ptr<Reuslt> find(const std::string input, int id = -1) const;
    std::shared_ptr<Reuslt> find(const std::vector<std::string> inputs, int id = -1) const;
    std::shared_ptr<Reuslt> find_next(const std::string input) const;
    std::shared_ptr<Reuslt> find_next(const std::vector<std::string> inputs) const;

private:
    std::vector<std::shared_ptr<const std::string>> m_commands;

    size_t m_count;

    void parser(std::string &&input);
};



/**
 * Option stores parameters, maximal and minimal.
 * When we put a value in it. It will adjust the value automatically.
 */

class Option {
private:
    enum class type {
        Invalid,
        String,
        Bool,
        Integer,
        Float,
    };

    type m_type{type::Invalid};
    std::string m_value{};
    int m_max{0};
    int m_min{0};

    Option(type t, std::string val, int max, int min) :
               m_type(t), m_value(val), m_max(max), m_min(min) {}

    operator int() const {
        assert(m_type == type::Integer);
        return std::stoi(m_value);
    }

    operator bool() const {
        assert(m_type == type::Bool);
        return (m_value == "true");
    }

    operator float() const {
        assert(m_type == type::Float);
        return std::stof(m_value);
    }

    operator std::string() const {
        assert(m_type == type::String);
        return m_value;
    }

    bool boundary_valid() const;

    template<typename T>
    void adjust();

    void option_handle() const;

public:
    Option() = default;

    void operator<<(const Option &&o) { *this = std::forward<decltype(o)>(o); }

    // Get Option object.
    template<typename T>
    static Option setoption(T val, int max = 0, int min = 0);

    // Get the value. We need to assign type.
    template<typename T>
    T get() const;

    // Set the value.
    template<typename T>
    void set(T value);
};

// Adjust the value. Be sure the value is not bigger 
// than maximal and smaller than minimal.
template<typename T>
void Option::adjust() {
    if (!boundary_valid()) {
        return;
    }

    const auto upper = static_cast<T>(m_max);
    const auto lower = static_cast<T>(m_min);
    const auto val = (T)*this;

    if (val > upper) {
        set<T>(upper);
    } else if (val < lower) {
        set<T>(lower);
    }
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
    float get_record_time(size_t) const;
    int get_record_count() const;
    const std::vector<float>& get_record() const;

private:
    std::chrono::steady_clock::time_point m_clock_time;

    std::vector<float> m_record;

    size_t record_count;
};

} // namespace Utils

#endif
