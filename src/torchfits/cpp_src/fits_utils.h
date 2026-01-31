#pragma once

#include <string>
#include <stdexcept>

namespace torchfits {

// Helper to validate FITS filenames to prevent command injection
// Rejects filenames starting or ending with |
void validate_fits_filename(const std::string& filename);

}
