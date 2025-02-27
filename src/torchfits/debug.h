#ifndef TORCHFITS_DEBUG_H
#define TORCHFITS_DEBUG_H

#ifdef DEBUG
    #include <iostream>
    #include <chrono>
    
    #define DEBUG_LOG(x) \
        do { \
            std::cerr << "[TORCHFITS:" << __func__ << ":" << __LINE__ << "] " << x << std::endl; \
            std::cerr.flush(); \
        } while (0)
    
    #define DEBUG_TENSOR(name, tensor) \
        do { \
            std::cerr << "[TORCHFITS:" << __func__ << ":" << __LINE__ << "] "; \
            std::cerr << name << ": size=" << tensor.sizes() \
                      << ", dtype=" << tensor.dtype() \
                      << ", device=" << tensor.device() << std::endl; \
            std::cerr.flush(); \
        } while (0)
    
    #define DEBUG_SCOPE \
        const auto debug_start = std::chrono::high_resolution_clock::now(); \
        const std::string _debug_func_name = __func__; \

#else
    #define DEBUG_LOG(x)
    #define DEBUG_TENSOR(name, tensor)
    #define DEBUG_SCOPE
#endif

#endif // TORCHFITS_DEBUG_H
