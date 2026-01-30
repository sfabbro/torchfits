#pragma once
#include <string>
#include <stdexcept>
#include <algorithm>

namespace torchfits {

// Security check: Prevent command injection via cfitsio pipe syntax
// This function should be called before passing any filename to cfitsio
inline void validate_fits_filename(const std::string& filename) {
    if (!filename.empty()) {
        size_t first = filename.find_first_not_of(" \t");
        size_t last = filename.find_last_not_of(" \t");

        if (first != std::string::npos) {
            // Check for leading/trailing pipe which triggers shell command execution in cfitsio
            if (filename[first] == '|' || filename[last] == '|') {
                 throw std::runtime_error("Security Error: Filenames starting or ending with '|' are not allowed to prevent command execution.");
            }
        }
    }
}

} // namespace torchfits
