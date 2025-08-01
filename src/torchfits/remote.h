#ifndef TORCHFITS_REMOTE_H
#define TORCHFITS_REMOTE_H

#include <string>
#include <fitsio.h>
#include <memory>

/**
 * @brief Remote file fetcher with HTTP range request support
 * 
 * This class provides efficient remote FITS file access with caching
 * and range request support for partial downloads.
 */
class RemoteFetcher {
public:
    /**
     * @brief Download and cache a remote FITS file
     * @param url Remote file URL
     * @param cache_dir Local cache directory
     * @return Path to local cached file
     */
    static std::string fetch_file(const std::string& url, const std::string& cache_dir = "");
    
    /**
     * @brief Check if a URL points to a remote file
     * @param filename_or_url File path or URL
     * @return true if remote, false if local
     */
    static bool is_remote(const std::string& filename_or_url);
    
    /**
     * @brief Get local file path, downloading if necessary
     * @param filename_or_url File path or URL
     * @return Local file path
     */
    static std::string ensure_local(const std::string& filename_or_url);

private:
    static std::string get_cache_dir();
    static std::string get_cached_filename(const std::string& url);
    static bool file_exists(const std::string& path);
    static void download_file(const std::string& url, const std::string& local_path);
};

#endif // TORCHFITS_REMOTE_H
