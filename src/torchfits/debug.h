#ifndef TORCHFITS_DEBUG_H
#define TORCHFITS_DEBUG_H

#include <iostream>
#include <chrono>
#include <string>

// Enhanced logging system that works in both debug and release builds
// with different verbosity levels

// Severity levels
enum class LogLevel {
    DEBUG,   // Detailed information (debug builds only)
    INFO,    // General information (both debug and release)
    WARNING, // Potential issues (both debug and release)
    ERROR    // Critical problems (both debug and release)
};

// Base logging function that works in all builds
inline void log_message(LogLevel level, const std::string& func, int line, const std::string& message) {
    const char* level_str = "";
    switch (level) {
        case LogLevel::DEBUG:   level_str = "DEBUG"; break;
        case LogLevel::INFO:    level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::ERROR:   level_str = "ERROR"; break;
    }
    
    std::cerr << "[TORCHFITS:" << level_str << ":" << func << ":" << line << "] " 
              << message << std::endl;
    std::cerr.flush();
}

// Debug-only macros (detailed info)
#ifdef DEBUG
    #define DEBUG_LOG(x) log_message(LogLevel::DEBUG, __func__, __LINE__, x)
    
    #define DEBUG_TENSOR(name, tensor) \
        do { \
            std::string tensor_info = std::string(name) + ": size=" + std::to_string(tensor.numel()) + \
                      ", dtype=" + std::to_string(static_cast<int>(tensor.dtype().toScalarType())) + \
                      ", device=" + tensor.device().str(); \
            log_message(LogLevel::DEBUG, __func__, __LINE__, tensor_info); \
        } while (0)
    
    #define DEBUG_SCOPE \
        const auto debug_start = std::chrono::high_resolution_clock::now(); \
        const std::string _debug_func_name = __func__; \
        auto debug_end_func = [&]() { \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - debug_start).count(); \
            log_message(LogLevel::DEBUG, _debug_func_name, __LINE__, \
                       "Function execution time: " + std::to_string(duration) + " μs"); \
        }; \
        std::unique_ptr<void, decltype(debug_end_func)*> debug_scope_guard(nullptr, debug_end_func);
#else
    #define DEBUG_LOG(x)
    #define DEBUG_TENSOR(name, tensor)
    #define DEBUG_SCOPE
#endif

// Release-level logging macros (always enabled)
#define INFO_LOG(x) log_message(LogLevel::INFO, __func__, __LINE__, x)
#define WARNING_LOG(x) log_message(LogLevel::WARNING, __func__, __LINE__, x)
#define ERROR_LOG(x) log_message(LogLevel::ERROR, __func__, __LINE__, x)

#endif // TORCHFITS_DEBUG_H
