#include "hardware.h"
#include <sys/stat.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdexcept>

namespace torchfits {

MMapHandle::MMapHandle(const std::string& filename) : MMapHandle(filename, false) {}

MMapHandle::MMapHandle(void* ptr, size_t size, int fd, bool owner)
    : ptr(ptr), size(size), fd(fd), owner(owner) {}

MMapHandle::MMapHandle(const std::string& filename, bool writable) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file descriptor: " + filename);
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        close(fd);
        throw std::runtime_error("Failed to stat file: " + filename);
    }
    size = st.st_size;

    int prot = PROT_READ | (writable ? PROT_WRITE : 0);
    ptr = mmap(nullptr, size, prot, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + filename);
    }
    owner = true;
}

void MMapHandle::cleanup() {
    if (ptr) {
        munmap(ptr, size);
        ptr = nullptr;
    }
    if (owner && fd != -1) {
        close(fd);
        fd = -1;
    }
    size = 0;
}

}  // namespace torchfits
