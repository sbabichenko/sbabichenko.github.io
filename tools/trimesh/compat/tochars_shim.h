// Emscripten 3.1's libc++ declares std::chars_format but implements no floating-point to_chars.
// The engine uses only the general format at a fixed precision, which is printf %.{p}g.
#pragma once
#include <charconv>
#include <cstdio>
#include <system_error>
namespace std {
inline to_chars_result to_chars(char* first, char* last, double v, chars_format, int precision) {
    const int n = std::snprintf(first, static_cast<size_t>(last - first), "%.*g", precision, v);
    if (n < 0 || n >= last - first) return {last, errc::value_too_large};
    return {first + n, errc{}};
}
}
