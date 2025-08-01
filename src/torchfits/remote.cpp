#include "remote.h"
#include "debug.h"
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <regex>

namespace fs = std::filesystem;

// Callback for writing downloaded data
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

bool RemoteFetcher::is_remote(const std::string& filename_or_url) {
    std::regex url_pattern(R"(^(https?|ftp|s3|gs|azure)://.+)");
    return std::regex_match(filename_or_url, url_pattern);
}

std::string RemoteFetcher::get_cache_dir() {
    const char* cache_env = std::getenv("TORCHFITS_CACHE_DIR");
    if (cache_env) {
        return std::string(cache_env);
    }
    
    const char* home = std::getenv("HOME");
    if (home) {
        return std::string(home) + "/.cache/torchfits";
    }
    
    return "/tmp/torchfits_cache";
}

std::string RemoteFetcher::get_cached_filename(const std::string& url) {
    // Create a safe filename from URL
    std::string safe_name = url;
    
    // Replace unsafe characters
    std::regex unsafe_chars(R"([/\\:*?"<>|])");
    safe_name = std::regex_replace(safe_name, unsafe_chars, "_");
    
    // Limit length and add extension if needed
    if (safe_name.length() > 200) {
        safe_name = safe_name.substr(0, 200);
    }
    
    if (safe_name.find(".fits") == std::string::npos) {
        safe_name += ".fits";
    }
    
    return safe_name;
}

bool RemoteFetcher::file_exists(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path);
}

void RemoteFetcher::download_file(const std::string& url, const std::string& local_path) {
    DEBUG_LOG("Downloading " + url + " to " + local_path);
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }
    
    // Create directory if needed
    fs::create_directories(fs::path(local_path).parent_path());
    
    std::ofstream file(local_path, std::ios::binary);
    if (!file) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to create local file: " + local_path);
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L); // 5 minute timeout
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "torchfits/0.1.0");
    
    CURLcode res = curl_easy_perform(curl);
    file.close();
    
    if (res != CURLE_OK) {
        fs::remove(local_path); // Clean up failed download
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to download file: " + std::string(curl_easy_strerror(res)));
    }
    
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);
    
    if (response_code != 200) {
        fs::remove(local_path);
        throw std::runtime_error("HTTP error " + std::to_string(response_code) + " downloading: " + url);
    }
    
    DEBUG_LOG("Successfully downloaded " + url);
}

std::string RemoteFetcher::fetch_file(const std::string& url, const std::string& cache_dir) {
    std::string cache_path = cache_dir.empty() ? get_cache_dir() : cache_dir;
    std::string local_filename = get_cached_filename(url);
    std::string local_path = cache_path + "/" + local_filename;
    
    if (file_exists(local_path)) {
        DEBUG_LOG("Using cached file: " + local_path);
        return local_path;
    }
    
    download_file(url, local_path);
    return local_path;
}

std::string RemoteFetcher::ensure_local(const std::string& filename_or_url) {
    if (is_remote(filename_or_url)) {
        return fetch_file(filename_or_url);
    }
    return filename_or_url;
}
