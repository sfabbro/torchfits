#pragma once
#include <string>
#include <stdexcept>

namespace torchfits {

inline void validate_fits_filename(const std::string& filename) {
    if (!filename.empty()) {
        size_t first = filename.find_first_not_of(" \t");
        size_t last = filename.find_last_not_of(" \t");

        if (first != std::string::npos) {
            if (filename[first] == '|' || filename[last] == '|') {
                 throw std::runtime_error("Security Error: Filenames starting or ending with '|' are not allowed to prevent command execution.");
            }
        }
    }
}

} // namespace torchfits
