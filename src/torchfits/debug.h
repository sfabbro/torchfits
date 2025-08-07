#ifndef TORCHFITS_DEBUG_H
#define TORCHFITS_DEBUG_H

#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

// Enhanced logging system that works in both debug and release builds
// with different verbosity levels

// Severity levels
enum class LogLevel {
    DEBUG_LEVEL,   // Detailed information (debug builds only)
    INFO,          // General information (both debug and release)
    WARNING,       // Potential issues (both debug and release)
    ERROR          // Critical problems (both debug and release)
};

// Base logging function that works in all builds
inline void log_message(LogLevel level, const std::string& func, int line, const std::string& message) {
    const char* level_str = "";
    switch (level) {
        case LogLevel::DEBUG_LEVEL: level_str = "DEBUG"; break;
        case LogLevel::INFO:        level_str = "INFO"; break;
        case LogLevel::WARNING:     level_str = "WARNING"; break;
        case LogLevel::ERROR:       level_str = "ERROR"; break;
    }
    
    std::cerr << "[TORCHFITS:" << level_str << ":" << func << ":" << line << "] " 
              << message << std::endl;
    std::cerr.flush();
}

// Debug-only macros (detailed info)
#ifdef DEBUG
    #define DEBUG_LOG(x) log_message(LogLevel::DEBUG_LEVEL, __func__, __LINE__, x)
    
    #define DEBUG_TENSOR(name, tensor) \
        do { \
            std::string tensor_info = std::string(name) + ": size=" + std::to_string(tensor.numel()) + \
                      ", dtype=" + std::to_string(static_cast<int>(tensor.dtype().toScalarType())) + \
                      ", device=" + tensor.device().str(); \
            log_message(LogLevel::DEBUG_LEVEL, __func__, __LINE__, tensor_info); \
        } while (0)
    
    // Helper class for RAII-based scope timing
    class DebugTimer {
    public:
        inline DebugTimer(const char* func_name, int line_num) : 
            start_time_(std::chrono::high_resolution_clock::now()), 
            func_name_(func_name), 
            line_num_(line_num) {}

        inline ~DebugTimer() {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
            std::stringstream ss;
            ss << "Function execution time: " << duration << " μs";
            log_message(LogLevel::DEBUG_LEVEL, func_name_, line_num_, ss.str());
        }

    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        const char* func_name_;
        int line_num_;
    };

    #define DEBUG_SCOPE DebugTimer debug_timer_instance(__func__, __LINE__)
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
