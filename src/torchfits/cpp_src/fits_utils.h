#pragma once
#include <string>
#include <stdexcept>

namespace torchfits {

// Security check: Prevent command injection via cfitsio pipe syntax
// cfitsio treats filenames ending/starting with '|' as pipes to shell commands.
// It also supports '!' prefix for overwrite.
// We must check the filename for '|' after stripping optional whitespace and '!' prefix.
inline void validate_fits_filename(const std::string& filename) {
    if (filename.empty()) return;

    std::string clean_name = filename;

    // Trim leading whitespace
    size_t first = clean_name.find_first_not_of(" \t");
    if (first == std::string::npos) return; // All whitespace

    // Trim trailing whitespace
    size_t last = clean_name.find_last_not_of(" \t");
    clean_name = clean_name.substr(first, last - first + 1);

    // Remove leading '!' which indicates overwrite in cfitsio
    if (!clean_name.empty() && clean_name.front() == '!') {
        clean_name.erase(0, 1);
        // Re-trim leading whitespace after '!'
        first = clean_name.find_first_not_of(" \t");
        if (first == std::string::npos) return;

        last = clean_name.find_last_not_of(" \t");
        clean_name = clean_name.substr(first, last - first + 1);
    }

    if (!clean_name.empty()) {
        if (clean_name.front() == '|' || clean_name.back() == '|') {
            throw std::runtime_error("Security Error: Filenames starting or ending with '|' are not allowed to prevent command execution.");
        }
    }
}

} // namespace torchfits
