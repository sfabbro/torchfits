#pragma once

#include <string>
#include <stdexcept>

namespace torchfits {

inline void check_fits_filename_security(const std::string& filename) {
    if (!filename.empty()) {
        size_t first = filename.find_first_not_of(" 	");
        size_t last = filename.find_last_not_of(" 	");

        if (first != std::string::npos) {
            size_t start_idx = first;

            // Allow multiple leading '!' because CFITSIO uses them for forced overwrite,
            // and skip spaces between '!' if any.
            while (start_idx != std::string::npos && filename[start_idx] == '!') {
                start_idx = filename.find_first_not_of(" 	", start_idx + 1);
            }

            if (start_idx != std::string::npos && filename[start_idx] == '|') {
                throw std::runtime_error("Security Error: Filenames starting with '|' are not allowed to prevent command execution.");
            }

            if (filename[last] == '|') {
                throw std::runtime_error("Security Error: Filenames ending with '|' are not allowed to prevent command execution.");
            }
        }
    }
}

} // namespace torchfits
