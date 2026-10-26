#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace torchfits {
namespace internal {

template <typename T>
inline T checked_nonnegative_add(T lhs, T rhs, const std::string& context) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_signed_v<T>) {
        if (lhs < 0 || rhs < 0) {
            throw std::runtime_error(context + ": negative value");
        }
    }
    if (lhs > std::numeric_limits<T>::max() - rhs) {
        throw std::runtime_error(context + ": addition overflow");
    }
    return static_cast<T>(lhs + rhs);
}

template <typename T>
inline T checked_nonnegative_multiply(T lhs, T rhs, const std::string& context) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_signed_v<T>) {
        if (lhs < 0 || rhs < 0) {
            throw std::runtime_error(context + ": negative value");
        }
    }
    if (rhs != 0 && lhs > std::numeric_limits<T>::max() / rhs) {
        throw std::runtime_error(context + ": multiplication overflow");
    }
    return static_cast<T>(lhs * rhs);
}

template <typename To, typename From>
inline To checked_nonnegative_cast(From value, const std::string& context) {
    static_assert(std::is_integral_v<To>);
    static_assert(std::is_integral_v<From>);
    if constexpr (std::is_signed_v<From>) {
        if (value < 0) {
            throw std::runtime_error(context + ": negative value");
        }
    }
    using UnsignedFrom = std::make_unsigned_t<From>;
    using UnsignedTo = std::make_unsigned_t<To>;
    const auto unsigned_value = static_cast<UnsignedFrom>(value);
    if constexpr (sizeof(UnsignedFrom) > sizeof(UnsignedTo) ||
                  (sizeof(UnsignedFrom) == sizeof(UnsignedTo) && std::is_signed_v<To>)) {
        if (unsigned_value > static_cast<UnsignedFrom>(std::numeric_limits<To>::max())) {
            throw std::runtime_error(context + ": conversion overflow");
        }
    }
    return static_cast<To>(value);
}

}
}
