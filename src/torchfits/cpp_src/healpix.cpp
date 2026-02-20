#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <unordered_map>

#include <ATen/Parallel.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include "torchfits_torch.h"

namespace torchfits {
namespace {

constexpr int64_t JRLL[12] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
constexpr int64_t JPLL[12] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
constexpr int64_t NB_XOFFSET[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
constexpr int64_t NB_YOFFSET[8] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int64_t NB_FACEARRAY[9][12] = {
    {8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9},
    {5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8},
    {-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1},
    {4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4},
    {-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1},
    {3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7},
    {2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3},
};
constexpr int64_t NB_SWAPARRAY[9][3] = {
    {0, 0, 3},
    {0, 0, 6},
    {0, 0, 0},
    {0, 0, 5},
    {0, 0, 0},
    {5, 0, 0},
    {0, 0, 0},
    {6, 0, 0},
    {3, 0, 0},
};
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 6.283185307179586476925286766559005768;
// Benchmark-relevant arrays are commonly O(1e5-1e6); parallelize earlier.
constexpr int64_t PARALLEL_MIN_ELEMS = 1 << 16;

inline bool is_valid_nside(int64_t nside) {
    return nside > 0 && (nside & (nside - 1)) == 0;
}

inline int64_t floor_div2(int64_t v) {
    return (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);
}

inline int64_t alm_size(int64_t lmax, int64_t mmax) {
    return (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) / 2;
}

inline int64_t ilog2_pow2(int64_t x) {
    int64_t s = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        ++s;
    }
    return s;
}

inline const std::array<uint8_t, 65536>& compact_lut16() {
    static const std::array<uint8_t, 65536> lut = []() {
        std::array<uint8_t, 65536> table{};
        for (uint32_t i = 0; i < 65536; ++i) {
            uint8_t v = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                v |= static_cast<uint8_t>(((i >> (2U * b)) & 1U) << b);
            }
            table[i] = v;
        }
        return table;
    }();
    return lut;
}

inline uint64_t isqrt_u64(uint64_t v) {
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(v)));
    while ((r + 1) * (r + 1) <= v) {
        ++r;
    }
    while (r * r > v) {
        --r;
    }
    return r;
}

inline uint64_t spread_bits_u64(uint64_t x) {
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;
    return x;
}

inline uint64_t compact_bits_u64(uint64_t x) {
    const auto& lut = compact_lut16();
    return static_cast<uint64_t>(lut[x & 0xFFFFULL])
           | (static_cast<uint64_t>(lut[(x >> 16) & 0xFFFFULL]) << 8)
           | (static_cast<uint64_t>(lut[(x >> 32) & 0xFFFFULL]) << 16)
           | (static_cast<uint64_t>(lut[(x >> 48) & 0xFFFFULL]) << 24);
}

inline double wrap_rad_2pi(double v) {
    double r = std::fmod(v, TWO_PI);
    if (r < 0.0) {
        r += TWO_PI;
    }
    return r;
}

inline double wrap_deg_360(double v) {
    double r = std::fmod(v, 360.0);
    if (r < 0.0) {
        r += 360.0;
    }
    return r;
}

inline int64_t xyf_to_nest(int64_t npface, int64_t ix, int64_t iy, int64_t face_num) {
    return face_num * npface
           + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(ix)))
           + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(iy)) << 1);
}

inline void nest_to_xyf(int64_t npface, int64_t pix_nest, int64_t& ix, int64_t& iy, int64_t& face_num) {
    face_num = pix_nest / npface;
    const uint64_t ipf = static_cast<uint64_t>(pix_nest % npface);
    ix = static_cast<int64_t>(compact_bits_u64(ipf));
    iy = static_cast<int64_t>(compact_bits_u64(ipf >> 1));
}

inline int64_t xyf_to_ring(
    int64_t nside,
    int64_t ncap,
    int64_t npix,
    int64_t nl4,
    int64_t ix,
    int64_t iy,
    int64_t face_num
) {
    const int64_t jr = JRLL[face_num] * nside - ix - iy - 1;
    int64_t nr;
    int64_t n_before;
    int64_t kshift = 0;
    if (jr < nside) {
        nr = jr;
        n_before = 2 * nr * (nr - 1);
    } else if (jr > (3 * nside)) {
        nr = nl4 - jr;
        n_before = npix - 2 * (nr + 1) * nr;
    } else {
        nr = nside;
        n_before = ncap + (jr - nside) * nl4;
        kshift = (jr - nside) & 1;
    }

    int64_t jp = (JPLL[face_num] * nr + ix - iy + 1 + kshift) / 2;
    if (jp > nl4) {
        jp -= nl4;
    }
    if (jp < 1) {
        jp += nl4;
    }
    return n_before + jp - 1;
}

inline void ring_to_xyf(
    int64_t nside,
    int64_t ncap,
    int64_t npix,
    int64_t nl2,
    int64_t nside_shift,
    int64_t four_nside_shift,
    int64_t four_nside_mask,
    int64_t p,
    int64_t& ix,
    int64_t& iy,
    int64_t& face_num
) {
    int64_t iring;
    int64_t iphi;
    int64_t kshift;
    int64_t nr;

    if (p < ncap) {
        iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(1 + 2 * p))) >> 1);
        iphi = (p + 1) - 2 * iring * (iring - 1);
        kshift = 0;
        nr = iring;
        face_num = (iphi - 1) / iring;
    } else if (p < (npix - ncap)) {
        const int64_t ip = p - ncap;
        iring = (ip >> four_nside_shift) + nside;
        iphi = (ip & four_nside_mask) + 1;
        kshift = (iring + nside) & 1;
        nr = nside;
        const int64_t ire = iring - nside + 1;
        const int64_t irm = nl2 + 2 - ire;
        const int64_t ifm = (iphi - (ire / 2) + nside - 1) >> nside_shift;
        const int64_t ifp = (iphi - (irm / 2) + nside - 1) >> nside_shift;
        if (ifp == ifm) {
            face_num = ifp | 4;
        } else if (ifp < ifm) {
            face_num = ifp;
        } else {
            face_num = ifm + 8;
        }
    } else {
        const int64_t ip = npix - p;
        const int64_t irs = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(2 * ip - 1))) >> 1);
        iphi = 4 * irs + 1 - (ip - 2 * irs * (irs - 1));
        iring = 2 * nl2 - irs;
        kshift = 0;
        nr = irs;
        face_num = 8 + (iphi - 1) / irs;
    }

    const int64_t irt = iring - JRLL[face_num] * nside + 1;
    int64_t ipt = 2 * iphi - JPLL[face_num] * nr - kshift - 1;
    if (ipt >= nl2) {
        ipt -= 8 * nside;
    }

    ix = floor_div2(ipt - irt);
    iy = floor_div2(-(ipt + irt));
}

inline int64_t ring_above_scalar(int64_t nside, double z) {
    const double az = std::abs(z);
    if (az <= (2.0 / 3.0)) {
        return static_cast<int64_t>(std::floor(nside * (2.0 - 1.5 * z)));
    }
    const double irp_f = nside * std::sqrt(std::max(0.0, 3.0 * (1.0 - az)));
    const int64_t irp = static_cast<int64_t>(std::floor(irp_f));
    if (z > 0.0) {
        return irp;
    }
    return 4 * nside - irp - 1;
}

inline void get_ring_info2_scalar(
    int64_t nside,
    int64_t ring,
    int64_t& startpix,
    int64_t& ringpix,
    double& theta,
    bool& shifted
) {
    const int64_t npix = 12 * nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const double fact2 = 4.0 / static_cast<double>(npix);
    const double fact1 = static_cast<double>(2 * nside) * fact2;

    const int64_t northr = (ring > (2 * nside)) ? (4 * nside - ring) : ring;
    if (northr < nside) {
        const double northf = static_cast<double>(northr);
        const double tmp = northf * northf * fact2;
        const double costheta = 1.0 - tmp;
        const double sintheta = std::sqrt(std::max(0.0, tmp * (2.0 - tmp)));
        theta = std::atan2(sintheta, costheta);
        ringpix = 4 * northr;
        shifted = true;
        startpix = 2 * northr * (northr - 1);
    } else {
        double costheta = (2.0 * static_cast<double>(nside) - static_cast<double>(northr)) * fact1;
        if (costheta > 1.0) {
            costheta = 1.0;
        } else if (costheta < -1.0) {
            costheta = -1.0;
        }
        theta = std::acos(costheta);
        ringpix = 4 * nside;
        shifted = (((northr - nside) & 1) == 0);
        startpix = ncap + (northr - nside) * ringpix;
    }

    if (northr != ring) {
        theta = PI - theta;
        startpix = npix - startpix - ringpix;
    }
}

struct RecurrenceCoeffs {
    double c1, c2_p, c3;
};

class RecurrenceCache {
    std::mutex mutex_;
    int64_t last_lmax_ = -1;
    int64_t last_mmax_ = -1;
    int64_t last_spin_ = -1;
    std::vector<RecurrenceCoeffs> coeffs_;
    std::vector<int64_t> m_offsets_;

public:
    const std::vector<RecurrenceCoeffs>& get_coeffs(int64_t lmax, int64_t mmax, int64_t spin, const std::vector<int64_t>& offsets) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lmax == last_lmax_ && mmax == last_mmax_ && spin == last_spin_) {
            return coeffs_;
        }

        int64_t total_size = 0;
        for (int64_t m = 0; m <= mmax; ++m) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            if (lmax >= l0 + 2) {
                total_size += (lmax - (l0 + 2) + 1);
            }
        }

        coeffs_.assign(total_size, {0, 0, 0});
        int64_t current_off = 0;
        for (int64_t m = 0; m <= mmax; ++m) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            for (int64_t l = l0 + 2; l <= lmax; ++l) {
                double kl = std::sqrt((double)(l * l - m * m) * (l * l - spin * spin));
                double preFactor = (double)l / kl;
                double c1 = preFactor * (2.0 * l - 1.0);
                double c3 = -preFactor * (std::sqrt((double)((l - 1) * (l - 1) - m * m) * ((l - 1) * (l - 1) - spin * spin)) / (l - 1));
                double shift = (double)(m * (-spin)) / (double)(l * (l - 1));
                double c2_p = -c1 * shift;
                coeffs_[current_off++] = {c1, c2_p, c3};
            }
        }
        last_lmax_ = lmax;
        last_mmax_ = mmax;
        last_spin_ = spin;
        return coeffs_;
    }
};

static RecurrenceCache g_recurrence_cache;

inline uint64_t ring_alias_cache_key(int64_t nph, int64_t mcount) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(nph)) << 32)
           | static_cast<uint64_t>(static_cast<uint32_t>(mcount));
}

class RingAliasIndexCache {
    std::mutex mutex_;
    std::unordered_map<uint64_t, torch::Tensor> plus_idx_;
    std::unordered_map<uint64_t, torch::Tensor> minus_idx_;
    std::unordered_map<uint64_t, torch::Tensor> both_idx_;
    static constexpr size_t kMaxEntries = 256;

    static torch::Tensor make_plus_idx(int64_t nph, int64_t mcount) {
        auto idx = torch::empty({mcount}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* ptr = idx.data_ptr<int64_t>();
        for (int64_t m = 0; m < mcount; ++m) {
            ptr[m] = m % nph;
        }
        return idx;
    }

    static torch::Tensor make_minus_idx(int64_t nph, int64_t mcount) {
        auto idx = torch::empty({mcount}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* ptr = idx.data_ptr<int64_t>();
        for (int64_t m = 0; m < mcount; ++m) {
            const int64_t k = m % nph;
            ptr[m] = (k == 0) ? 0 : (nph - k);
        }
        return idx;
    }

    static torch::Tensor make_both_idx(int64_t nph, int64_t mcount) {
        auto idx = torch::empty({2 * mcount}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* ptr = idx.data_ptr<int64_t>();
        for (int64_t m = 0; m < mcount; ++m) {
            const int64_t k = m % nph;
            ptr[m] = k;
            ptr[mcount + m] = (k == 0) ? 0 : (nph - k);
        }
        return idx;
    }

public:
    const torch::Tensor& plus(int64_t nph, int64_t mcount) {
        const uint64_t key = ring_alias_cache_key(nph, mcount);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = plus_idx_.find(key);
        if (it != plus_idx_.end()) {
            return it->second;
        }
        if (plus_idx_.size() > kMaxEntries) {
            plus_idx_.clear();
        }
        return plus_idx_.emplace(key, make_plus_idx(nph, mcount)).first->second;
    }

    const torch::Tensor& minus(int64_t nph, int64_t mcount) {
        const uint64_t key = ring_alias_cache_key(nph, mcount);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = minus_idx_.find(key);
        if (it != minus_idx_.end()) {
            return it->second;
        }
        if (minus_idx_.size() > kMaxEntries) {
            minus_idx_.clear();
        }
        return minus_idx_.emplace(key, make_minus_idx(nph, mcount)).first->second;
    }

    const torch::Tensor& both(int64_t nph, int64_t mcount) {
        const uint64_t key = ring_alias_cache_key(nph, mcount);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = both_idx_.find(key);
        if (it != both_idx_.end()) {
            return it->second;
        }
        if (both_idx_.size() > kMaxEntries) {
            both_idx_.clear();
        }
        return both_idx_.emplace(key, make_both_idx(nph, mcount)).first->second;
    }
};

static RingAliasIndexCache g_ring_alias_idx_cache;

inline bool spin_mlim_prune_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("TORCHFITS_SPIN_MLIM_PRUNE");
        if (!env) {
            return true;
        }
        return env[0] != '0';
    }();
    return enabled;
}

inline int64_t sharp_mlim_estimate(int64_t lmax, int64_t spin_abs, double sth, double cth) {
    double ofs = static_cast<double>(lmax) * 0.01;
    if (ofs < 100.0) {
        ofs = 100.0;
    }
    const double b = -2.0 * static_cast<double>(spin_abs) * std::abs(cth);
    const double t1 = static_cast<double>(lmax) * sth + ofs;
    const double c = static_cast<double>(spin_abs * spin_abs) - t1 * t1;
    const double discr = b * b - 4.0 * c;
    if (discr <= 0.0) {
        return lmax;
    }
    double res = (-b + std::sqrt(discr)) * 0.5;
    if (res < 0.0) {
        res = 0.0;
    }
    if (res > static_cast<double>(lmax)) {
        res = static_cast<double>(lmax);
    }
    return static_cast<int64_t>(res + 0.5);
}

}  // namespace

torch::Tensor healpix_ang2pix_ring_cpu(int64_t nside, const torch::Tensor& ra_deg, const torch::Tensor& dec_deg) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!ra_deg.device().is_cpu() || !dec_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_ang2pix_ring_cpu expects CPU tensors");
    }
    if (ra_deg.scalar_type() != torch::kDouble || dec_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_ang2pix_ring_cpu expects float64 tensors");
    }
    if (!ra_deg.sizes().equals(dec_deg.sizes())) {
        throw std::runtime_error("RA/Dec shape mismatch");
    }

    auto ra = ra_deg.contiguous().reshape({-1});
    auto dec = dec_deg.contiguous().reshape({-1});
    auto out = torch::empty_like(ra, torch::TensorOptions().dtype(torch::kInt64));
    const auto* ra_ptr = ra.data_ptr<double>();
    const auto* dec_ptr = dec.data_ptr<double>();
    auto* dst = out.data_ptr<int64_t>();
    const int64_t n = ra.numel();
    const int64_t npix = 12 * nside * nside;

    for (int64_t i = 0; i < n; ++i) {
        const double z = std::sin(dec_ptr[i] * PI / 180.0);
        const double za = std::abs(z);
        const double tt = wrap_rad_2pi(ra_ptr[i] * PI / 180.0) * (2.0 / PI);

        int64_t pix;
        if (za <= (2.0 / 3.0)) {
            const double temp1 = nside * (0.5 + tt);
            const double temp2 = nside * z * 0.75;
            const int64_t jp = static_cast<int64_t>(temp1 - temp2);
            const int64_t jm = static_cast<int64_t>(temp1 + temp2);
            const int64_t ir = nside + 1 + jp - jm;
            const int64_t kshift = 1 - (ir & 1);
            int64_t ip = ((jp + jm - nside + kshift + 1) / 2) + 1;
            if (ip > (4 * nside)) {
                ip -= (4 * nside);
            }
            if (ip < 1) {
                ip += (4 * nside);
            }
            pix = 2 * nside * (nside - 1) + (ir - 1) * (4 * nside) + (ip - 1);
        } else {
            const int64_t ntt = static_cast<int64_t>(tt);
            const double tp = tt - static_cast<double>(ntt);
            const double tmp = nside * std::sqrt(3.0 * (1.0 - std::abs(z)));
            const int64_t jp = static_cast<int64_t>(tp * tmp);
            const int64_t jm = static_cast<int64_t>((1.0 - tp) * tmp);
            const int64_t ir = jp + jm + 1;
            const int64_t ip = static_cast<int64_t>(tt * static_cast<double>(ir)) % (4 * ir);
            if (z > 0.0) {
                pix = 2 * ir * (ir - 1) + ip;
            } else {
                pix = npix - 2 * ir * (ir + 1) + ip;
            }
        }
        dst[i] = pix;
    }
    return out.reshape(ra_deg.sizes().vec());
}

torch::Tensor healpix_ang2pix_nested_cpu(int64_t nside, const torch::Tensor& ra_deg, const torch::Tensor& dec_deg) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!ra_deg.device().is_cpu() || !dec_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_ang2pix_nested_cpu expects CPU tensors");
    }
    if (ra_deg.scalar_type() != torch::kDouble || dec_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_ang2pix_nested_cpu expects float64 tensors");
    }
    if (!ra_deg.sizes().equals(dec_deg.sizes())) {
        throw std::runtime_error("RA/Dec shape mismatch");
    }

    auto ra = ra_deg.contiguous().reshape({-1});
    auto dec = dec_deg.contiguous().reshape({-1});
    auto out = torch::empty_like(ra, torch::TensorOptions().dtype(torch::kInt64));
    const auto* ra_ptr = ra.data_ptr<double>();
    const auto* dec_ptr = dec.data_ptr<double>();
    auto* dst = out.data_ptr<int64_t>();
    const int64_t n = ra.numel();
    const int64_t npface = nside * nside;
    const int64_t nmask = nside - 1;

    for (int64_t i = 0; i < n; ++i) {
        const double z = std::sin(dec_ptr[i] * PI / 180.0);
        const double za = std::abs(z);
        const double tt = wrap_rad_2pi(ra_ptr[i] * PI / 180.0) * (2.0 / PI);

        int64_t face_num;
        int64_t ix;
        int64_t iy;

        if (za <= (2.0 / 3.0)) {
            const double temp1 = nside * (0.5 + tt);
            const double temp2 = nside * (z * 0.75);
            const int64_t jp = static_cast<int64_t>(temp1 - temp2);
            const int64_t jm = static_cast<int64_t>(temp1 + temp2);
            const int64_t ifp = jp / nside;
            const int64_t ifm = jm / nside;
            if (ifp == ifm) {
                face_num = ifp | 4;
            } else if (ifp < ifm) {
                face_num = ifp;
            } else {
                face_num = ifm + 8;
            }
            ix = jm & nmask;
            iy = nside - (jp & nmask) - 1;
        } else {
            int64_t ntt = static_cast<int64_t>(tt);
            if (ntt >= 4) {
                ntt = 3;
            }
            const double tp = tt - static_cast<double>(ntt);
            const double tmp = nside * std::sqrt(3.0 * (1.0 - std::abs(z)));
            int64_t jp = static_cast<int64_t>(tp * tmp);
            int64_t jm = static_cast<int64_t>((1.0 - tp) * tmp);
            if (jp > (nside - 1)) {
                jp = nside - 1;
            }
            if (jm > (nside - 1)) {
                jm = nside - 1;
            }
            if (z >= 0.0) {
                face_num = ntt;
                ix = nside - jm - 1;
                iy = nside - jp - 1;
            } else {
                face_num = ntt + 8;
                ix = jp;
                iy = jm;
            }
        }

        dst[i] = face_num * npface
                 + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(ix)))
                 + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(iy)) << 1);
    }
    return out.reshape(ra_deg.sizes().vec());
}

std::pair<torch::Tensor, torch::Tensor> healpix_pix2ang_ring_cpu(int64_t nside, const torch::Tensor& pix_ring) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_ring.device().is_cpu()) {
        throw std::runtime_error("healpix_pix2ang_ring_cpu expects CPU tensor");
    }
    if (pix_ring.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_pix2ang_ring_cpu expects int64 tensor");
    }

    auto pix = pix_ring.contiguous().reshape({-1});
    auto ra = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));
    auto dec = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));

    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const double fact2 = 4.0 / static_cast<double>(npix);

    const int64_t* in = pix.data_ptr<int64_t>();
    double* ra_ptr = ra.data_ptr<double>();
    double* dec_ptr = dec.data_ptr<double>();
    const int64_t n = pix.numel();
    const int64_t min_pix = pix.min().item<int64_t>();
    const int64_t max_pix = pix.max().item<int64_t>();
    if (min_pix < 0 || max_pix >= npix) {
        throw std::runtime_error("pixel index out of range for nside");
    }
    const double fact1 = static_cast<double>(2 * nside) * fact2;
    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];

            double z;
            double phi;
            if (p < ncap) {
                const int64_t iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(1 + 2 * p))) >> 1);
                const int64_t iphi = (p + 1) - 2 * iring * (iring - 1);
                z = 1.0 - static_cast<double>(iring * iring) * fact2;
                phi = (static_cast<double>(iphi) - 0.5) * ((PI / 2.0) / static_cast<double>(iring));
            } else if (p < (npix - ncap)) {
                const int64_t ip = p - ncap;
                const int64_t iring = (ip / (4 * nside)) + nside;
                const int64_t iphi = (ip % (4 * nside)) + 1;
                const double fodd = (((iring + nside) & 1) != 0) ? 1.0 : 0.5;
                z = static_cast<double>(2 * nside - iring) * fact1;
                phi = (static_cast<double>(iphi) - fodd) * (PI / static_cast<double>(2 * nside));
            } else {
                const int64_t ip = npix - p;
                const int64_t iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(2 * ip - 1))) >> 1);
                const int64_t iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
                z = -1.0 + static_cast<double>(iring * iring) * fact2;
                phi = (static_cast<double>(iphi) - 0.5) * ((PI / 2.0) / static_cast<double>(iring));
            }

            if (z > 1.0) {
                z = 1.0;
            } else if (z < -1.0) {
                z = -1.0;
            }
            ra_ptr[i] = wrap_deg_360(phi * 180.0 / PI);
            dec_ptr[i] = std::asin(z) * 180.0 / PI;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 4096, compute_block);
    } else {
        compute_block(0, n);
    }

    auto shape = pix_ring.sizes().vec();
    return {ra.reshape(shape), dec.reshape(shape)};
}

std::pair<torch::Tensor, torch::Tensor> healpix_pix2ang_nested_cpu(int64_t nside, const torch::Tensor& pix_nest) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_nest.device().is_cpu()) {
        throw std::runtime_error("healpix_pix2ang_nested_cpu expects CPU tensor");
    }
    if (pix_nest.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_pix2ang_nested_cpu expects int64 tensor");
    }

    auto pix = pix_nest.contiguous().reshape({-1});
    auto ra = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));
    auto dec = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));

    const int64_t npface = nside * nside;
    const int64_t npix = 12 * nside * nside;
    const int64_t nl4 = 4 * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const double fact2 = 4.0 / static_cast<double>(npix);
    const double fact1 = static_cast<double>(2 * nside) * fact2;

    const int64_t* in = pix.data_ptr<int64_t>();
    double* ra_ptr = ra.data_ptr<double>();
    double* dec_ptr = dec.data_ptr<double>();
    const int64_t n = pix.numel();
    const int64_t min_pix = pix.min().item<int64_t>();
    const int64_t max_pix = pix.max().item<int64_t>();
    if (min_pix < 0 || max_pix >= npix) {
        throw std::runtime_error("pixel index out of range for nside");
    }
    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            const int64_t face_num = p / npface;
            const uint64_t ipf = static_cast<uint64_t>(p % npface);
            const int64_t ix = static_cast<int64_t>(compact_bits_u64(ipf));
            const int64_t iy = static_cast<int64_t>(compact_bits_u64(ipf >> 1));
            const int64_t jr = JRLL[face_num] * nside - ix - iy - 1;

            int64_t nr;
            int64_t kshift = 0;
            double z;
            if (jr < nside) {
                nr = jr;
                z = 1.0 - static_cast<double>(nr * nr) * fact2;
            } else if (jr > (3 * nside)) {
                nr = nl4 - jr;
                z = static_cast<double>(nr * nr) * fact2 - 1.0;
            } else {
                nr = nside;
                z = static_cast<double>(2 * nside - jr) * fact1;
                kshift = (jr - nside) & 1;
            }

            int64_t jp = (JPLL[face_num] * nr + ix - iy + 1 + kshift) / 2;
            if (jp > nl4) {
                jp -= nl4;
            }
            if (jp < 1) {
                jp += nl4;
            }

            const double phi = (static_cast<double>(jp) - 0.5 * (static_cast<double>(kshift) + 1.0))
                               * ((PI / 2.0) / static_cast<double>(nr));
            if (z > 1.0) {
                z = 1.0;
            } else if (z < -1.0) {
                z = -1.0;
            }
            double ra = phi * 180.0 / PI;
            if (ra >= 360.0) {
                ra -= 360.0;
            } else if (ra < 0.0) {
                ra += 360.0;
            }
            ra_ptr[i] = ra;
            dec_ptr[i] = std::asin(z) * 180.0 / PI;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 8192, compute_block);
    } else {
        compute_block(0, n);
    }

    auto shape = pix_nest.sizes().vec();
    return {ra.reshape(shape), dec.reshape(shape)};
}

torch::Tensor healpix_ring2nest_cpu(int64_t nside, const torch::Tensor& pix_ring) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_ring.device().is_cpu()) {
        throw std::runtime_error("healpix_ring2nest_cpu expects CPU tensor");
    }
    if (pix_ring.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring2nest_cpu expects int64 tensor");
    }

    auto pix = pix_ring.contiguous();
    auto out = torch::empty_like(pix);

    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const int64_t nl2 = 2 * nside;
    const int64_t nside_shift = ilog2_pow2(nside);
    const int64_t four_nside_shift = nside_shift + 2;
    const int64_t four_nside_mask = (4 * nside) - 1;

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t n = pix.numel();
    if (n == 0) {
        return out;
    }
    std::vector<uint64_t> spread_x(static_cast<size_t>(nside));
    std::vector<uint64_t> spread_y(static_cast<size_t>(nside));
    for (int64_t i = 0; i < nside; ++i) {
        spread_x[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i));
        spread_y[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i)) << 1;
    }
    std::atomic<bool> invalid{false};

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            if (p < 0 || p >= npix) {
                invalid.store(true, std::memory_order_relaxed);
                dst[i] = -1;
                continue;
            }
            int64_t iring;
            int64_t iphi;
            int64_t kshift;
            int64_t nr;
            int64_t face_num;

            if (p < ncap) {
                iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(1 + 2 * p))) >> 1);
                iphi = (p + 1) - 2 * iring * (iring - 1);
                kshift = 0;
                nr = iring;
                face_num = (iphi - 1) / iring;
            } else if (p < (npix - ncap)) {
                const int64_t ip = p - ncap;
                iring = (ip >> four_nside_shift) + nside;
                iphi = (ip & four_nside_mask) + 1;
                kshift = (iring + nside) & 1;
                nr = nside;
                const int64_t ire = iring - nside + 1;
                const int64_t irm = nl2 + 2 - ire;
                const int64_t ifm = (iphi - (ire / 2) + nside - 1) >> nside_shift;
                const int64_t ifp = (iphi - (irm / 2) + nside - 1) >> nside_shift;
                if (ifp == ifm) {
                    face_num = ifp | 4;
                } else if (ifp < ifm) {
                    face_num = ifp;
                } else {
                    face_num = ifm + 8;
                }
            } else {
                const int64_t ip = npix - p;
                const int64_t irs = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(2 * ip - 1))) >> 1);
                iphi = 4 * irs + 1 - (ip - 2 * irs * (irs - 1));
                iring = 2 * nl2 - irs;
                kshift = 0;
                nr = irs;
                face_num = 8 + (iphi - 1) / irs;
            }

            const int64_t irt = iring - JRLL[face_num] * nside + 1;
            int64_t ipt = 2 * iphi - JPLL[face_num] * nr - kshift - 1;
            if (ipt >= nl2) {
                ipt -= 8 * nside;
            }

            const int64_t ix = floor_div2(ipt - irt);
            const int64_t iy = floor_div2(-(ipt + irt));
            if (ix < 0 || ix >= nside || iy < 0 || iy >= nside) {
                invalid.store(true, std::memory_order_relaxed);
                dst[i] = -1;
                continue;
            }
            dst[i] = face_num * npface
                     + static_cast<int64_t>(spread_x[static_cast<size_t>(ix)])
                     + static_cast<int64_t>(spread_y[static_cast<size_t>(iy)]);
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 16384, compute_block);
    } else {
        compute_block(0, n);
    }
    if (invalid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    return out;
}

torch::Tensor healpix_nest2ring_cpu(int64_t nside, const torch::Tensor& pix_nest) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_nest.device().is_cpu()) {
        throw std::runtime_error("healpix_nest2ring_cpu expects CPU tensor");
    }
    if (pix_nest.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_nest2ring_cpu expects int64 tensor");
    }

    auto pix = pix_nest.contiguous();
    auto out = torch::empty_like(pix);

    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const int64_t nl4 = 4 * nside;

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t n = pix.numel();
    const int64_t min_pix = pix.min().item<int64_t>();
    const int64_t max_pix = pix.max().item<int64_t>();
    if (min_pix < 0 || max_pix >= npix) {
        throw std::runtime_error("pixel index out of range for nside");
    }
    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            const int64_t face_num = p / npface;
            const uint64_t ipf = static_cast<uint64_t>(p % npface);
            const int64_t ix = static_cast<int64_t>(compact_bits_u64(ipf));
            const int64_t iy = static_cast<int64_t>(compact_bits_u64(ipf >> 1));

            const int64_t jr = JRLL[face_num] * nside - ix - iy - 1;
            int64_t nr;
            int64_t n_before;
            int64_t kshift = 0;
            if (jr < nside) {
                nr = jr;
                n_before = 2 * nr * (nr - 1);
            } else if (jr > (3 * nside)) {
                nr = nl4 - jr;
                n_before = npix - 2 * (nr + 1) * nr;
            } else {
                nr = nside;
                n_before = ncap + (jr - nside) * nl4;
                kshift = (jr - nside) & 1;
            }

            int64_t jp = (JPLL[face_num] * nr + ix - iy + 1 + kshift) / 2;
            if (jp > nl4) {
                jp -= nl4;
            }
            if (jp < 1) {
                jp += nl4;
            }
            dst[i] = n_before + jp - 1;
        }
    };
    at::parallel_for(0, n, 16384, compute_block);

    return out;
}

static torch::Tensor healpix_neighbors_cpu_impl(
    int64_t nside,
    const torch::Tensor& pix_in,
    bool input_nest,
    bool output_nest
) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_in.device().is_cpu()) {
        throw std::runtime_error("healpix_neighbors_cpu expects CPU tensor");
    }
    if (pix_in.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_neighbors_cpu expects int64 tensor");
    }

    auto pix = pix_in.contiguous().reshape({-1});
    const int64_t n = pix.numel();
    auto out = torch::full({n, 8}, static_cast<int64_t>(-1), torch::TensorOptions().dtype(torch::kInt64));
    if (n == 0) {
        auto shape = pix_in.sizes().vec();
        shape.push_back(8);
        return out.reshape(shape);
    }

    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const int64_t nl2 = 2 * nside;
    const int64_t nl4 = 4 * nside;
    const int64_t nside_shift = ilog2_pow2(nside);
    const int64_t four_nside_shift = nside_shift + 2;
    const int64_t four_nside_mask = (4 * nside) - 1;
    const int64_t nsm1 = nside - 1;
    std::vector<uint64_t> spread_x;
    std::vector<uint64_t> spread_y;
    if (output_nest) {
        spread_x.resize(static_cast<size_t>(nside));
        spread_y.resize(static_cast<size_t>(nside));
        for (int64_t i = 0; i < nside; ++i) {
            spread_x[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i));
            spread_y[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i)) << 1;
        }
    }

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t min_pix = pix.min().item<int64_t>();
    const int64_t max_pix = pix.max().item<int64_t>();
    if (min_pix < 0 || max_pix >= npix) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            int64_t ix;
            int64_t iy;
            int64_t face_num;
            if (input_nest) {
                nest_to_xyf(npface, p, ix, iy, face_num);
            } else {
                ring_to_xyf(
                    nside,
                    ncap,
                    npix,
                    nl2,
                    nside_shift,
                    four_nside_shift,
                    four_nside_mask,
                    p,
                    ix,
                    iy,
                    face_num
                );
            }

            const int64_t row_off = i * 8;
            if (ix > 0 && ix < nsm1 && iy > 0 && iy < nsm1) {
                if (output_nest) {
                    const uint64_t face_base = static_cast<uint64_t>(face_num * npface);
                    const uint64_t sxm = spread_x[static_cast<size_t>(ix - 1)];
                    const uint64_t sx0 = spread_x[static_cast<size_t>(ix)];
                    const uint64_t sxp = spread_x[static_cast<size_t>(ix + 1)];
                    const uint64_t sym = spread_y[static_cast<size_t>(iy - 1)];
                    const uint64_t sy0 = spread_y[static_cast<size_t>(iy)];
                    const uint64_t syp = spread_y[static_cast<size_t>(iy + 1)];

                    dst[row_off + 0] = static_cast<int64_t>(face_base + sxm + sy0); // SW
                    dst[row_off + 1] = static_cast<int64_t>(face_base + sxm + syp); // W
                    dst[row_off + 2] = static_cast<int64_t>(face_base + sx0 + syp); // NW
                    dst[row_off + 3] = static_cast<int64_t>(face_base + sxp + syp); // N
                    dst[row_off + 4] = static_cast<int64_t>(face_base + sxp + sy0); // NE
                    dst[row_off + 5] = static_cast<int64_t>(face_base + sxp + sym); // E
                    dst[row_off + 6] = static_cast<int64_t>(face_base + sx0 + sym); // SE
                    dst[row_off + 7] = static_cast<int64_t>(face_base + sxm + sym); // S
                } else {
                    dst[row_off + 0] = xyf_to_ring(nside, ncap, npix, nl4, ix - 1, iy, face_num);
                    dst[row_off + 1] = xyf_to_ring(nside, ncap, npix, nl4, ix - 1, iy + 1, face_num);
                    dst[row_off + 2] = xyf_to_ring(nside, ncap, npix, nl4, ix, iy + 1, face_num);
                    dst[row_off + 3] = xyf_to_ring(nside, ncap, npix, nl4, ix + 1, iy + 1, face_num);
                    dst[row_off + 4] = xyf_to_ring(nside, ncap, npix, nl4, ix + 1, iy, face_num);
                    dst[row_off + 5] = xyf_to_ring(nside, ncap, npix, nl4, ix + 1, iy - 1, face_num);
                    dst[row_off + 6] = xyf_to_ring(nside, ncap, npix, nl4, ix, iy - 1, face_num);
                    dst[row_off + 7] = xyf_to_ring(nside, ncap, npix, nl4, ix - 1, iy - 1, face_num);
                }
                continue;
            }

            for (int64_t m = 0; m < 8; ++m) {
                int64_t x = ix + NB_XOFFSET[m];
                int64_t y = iy + NB_YOFFSET[m];
                int64_t f = face_num;

                if (x < 0 || x > nsm1 || y < 0 || y > nsm1) {
                    int64_t nbnum = 4;
                    if (x < 0) {
                        x += nside;
                        nbnum -= 1;
                    } else if (x >= nside) {
                        x -= nside;
                        nbnum += 1;
                    }
                    if (y < 0) {
                        y += nside;
                        nbnum -= 3;
                    } else if (y >= nside) {
                        y -= nside;
                        nbnum += 3;
                    }

                    f = NB_FACEARRAY[nbnum][face_num];
                    if (f < 0) {
                        dst[row_off + m] = -1;
                        continue;
                    }

                    const int64_t bits = NB_SWAPARRAY[nbnum][face_num >> 2];
                    if ((bits & 1) != 0) {
                        x = nside - x - 1;
                    }
                    if ((bits & 2) != 0) {
                        y = nside - y - 1;
                    }
                    if ((bits & 4) != 0) {
                        const int64_t t = x;
                        x = y;
                        y = t;
                    }
                }

                if (output_nest) {
                    dst[row_off + m] = xyf_to_nest(npface, x, y, f);
                } else {
                    dst[row_off + m] = xyf_to_ring(nside, ncap, npix, nl4, x, y, f);
                }
            }
        }
    };

    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 8192, compute_block);
    } else {
        compute_block(0, n);
    }

    auto shape = pix_in.sizes().vec();
    shape.push_back(8);
    return out.reshape(shape);
}

static torch::Tensor healpix_neighbors_nested_cpu_impl(int64_t nside, const torch::Tensor& pix_in) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_in.device().is_cpu()) {
        throw std::runtime_error("healpix_neighbors_cpu expects CPU tensor");
    }
    if (pix_in.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_neighbors_cpu expects int64 tensor");
    }

    auto pix = pix_in.contiguous().reshape({-1});
    const int64_t n = pix.numel();
    auto out = torch::full({n, 8}, static_cast<int64_t>(-1), torch::TensorOptions().dtype(torch::kInt64));
    if (n == 0) {
        auto shape = pix_in.sizes().vec();
        shape.push_back(8);
        return out.reshape(shape);
    }

    const int64_t npface = nside * nside;
    const int64_t npix = 12 * nside * nside;
    const int64_t nsm1 = nside - 1;
    std::vector<uint64_t> spread_x(static_cast<size_t>(nside));
    std::vector<uint64_t> spread_y(static_cast<size_t>(nside));
    for (int64_t i = 0; i < nside; ++i) {
        spread_x[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i));
        spread_y[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i)) << 1;
    }

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t min_pix = pix.min().item<int64_t>();
    const int64_t max_pix = pix.max().item<int64_t>();
    if (min_pix < 0 || max_pix >= npix) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            int64_t ix;
            int64_t iy;
            int64_t face_num;
            nest_to_xyf(npface, p, ix, iy, face_num);

            const int64_t row_off = i * 8;
            if (ix > 0 && ix < nsm1 && iy > 0 && iy < nsm1) {
                // Interior pixels stay on the same face; skip boundary machinery.
                const uint64_t face_base = static_cast<uint64_t>(face_num * npface);
                const uint64_t sxm = spread_x[static_cast<size_t>(ix - 1)];
                const uint64_t sx0 = spread_x[static_cast<size_t>(ix)];
                const uint64_t sxp = spread_x[static_cast<size_t>(ix + 1)];
                const uint64_t sym = spread_y[static_cast<size_t>(iy - 1)];
                const uint64_t sy0 = spread_y[static_cast<size_t>(iy)];
                const uint64_t syp = spread_y[static_cast<size_t>(iy + 1)];

                dst[row_off + 0] = static_cast<int64_t>(face_base + sxm + sy0); // SW
                dst[row_off + 1] = static_cast<int64_t>(face_base + sxm + syp); // W
                dst[row_off + 2] = static_cast<int64_t>(face_base + sx0 + syp); // NW
                dst[row_off + 3] = static_cast<int64_t>(face_base + sxp + syp); // N
                dst[row_off + 4] = static_cast<int64_t>(face_base + sxp + sy0); // NE
                dst[row_off + 5] = static_cast<int64_t>(face_base + sxp + sym); // E
                dst[row_off + 6] = static_cast<int64_t>(face_base + sx0 + sym); // SE
                dst[row_off + 7] = static_cast<int64_t>(face_base + sxm + sym); // S
                continue;
            }

            const int64_t band = face_num >> 2;
            for (int64_t m = 0; m < 8; ++m) {
                int64_t x = ix + NB_XOFFSET[m];
                int64_t y = iy + NB_YOFFSET[m];
                int64_t f = face_num;

                if (x < 0 || x > nsm1 || y < 0 || y > nsm1) {
                    int64_t nbnum = 4;
                    if (x < 0) {
                        x += nside;
                        nbnum -= 1;
                    } else if (x >= nside) {
                        x -= nside;
                        nbnum += 1;
                    }
                    if (y < 0) {
                        y += nside;
                        nbnum -= 3;
                    } else if (y >= nside) {
                        y -= nside;
                        nbnum += 3;
                    }

                    f = NB_FACEARRAY[nbnum][face_num];
                    if (f < 0) {
                        dst[row_off + m] = -1;
                        continue;
                    }

                    const int64_t bits = NB_SWAPARRAY[nbnum][band];
                    if ((bits & 1) != 0) {
                        x = nside - x - 1;
                    }
                    if ((bits & 2) != 0) {
                        y = nside - y - 1;
                    }
                    if ((bits & 4) != 0) {
                        const int64_t t = x;
                        x = y;
                        y = t;
                    }
                }
                dst[row_off + m] = static_cast<int64_t>(
                    static_cast<uint64_t>(f * npface) + spread_x[static_cast<size_t>(x)] + spread_y[static_cast<size_t>(y)]
                );
            }
        }
    };

    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 8192, compute_block);
    } else {
        compute_block(0, n);
    }

    auto shape = pix_in.sizes().vec();
    shape.push_back(8);
    return out.reshape(shape);
}

torch::Tensor healpix_neighbors_ring_cpu(int64_t nside, const torch::Tensor& pix_ring) {
    return healpix_neighbors_cpu_impl(nside, pix_ring, false, false);
}

torch::Tensor healpix_neighbors_nested_cpu(int64_t nside, const torch::Tensor& pix_nest) {
    return healpix_neighbors_nested_cpu_impl(nside, pix_nest);
}

torch::Tensor healpix_scalar_alm2map_direct_cpu(
    const torch::Tensor& alm_values,
    const torch::Tensor& cos_theta,
    const torch::Tensor& phi,
    int64_t lmax,
    int64_t mmax
) {
    if (!alm_values.device().is_cpu() || !cos_theta.device().is_cpu() || !phi.device().is_cpu()) {
        throw std::runtime_error("healpix_scalar_alm2map_direct_cpu expects CPU tensors");
    }
    if (alm_values.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_scalar_alm2map_direct_cpu expects complex128 alm tensor");
    }
    if (cos_theta.scalar_type() != torch::kDouble || phi.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_scalar_alm2map_direct_cpu expects float64 cos_theta/phi tensors");
    }
    if (lmax < 0) {
        throw std::runtime_error("lmax must be non-negative");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("mmax must satisfy 0 <= mmax <= lmax");
    }

    torch::Tensor rows;
    if (alm_values.dim() == 1) {
        rows = alm_values.contiguous().reshape({1, alm_values.numel()});
    } else if (alm_values.dim() == 2) {
        rows = alm_values.contiguous();
    } else {
        throw std::runtime_error("alm_values must be rank-1 or rank-2");
    }

    auto x = cos_theta.contiguous().reshape({-1});
    auto ph = phi.contiguous().reshape({-1});
    if (x.numel() != ph.numel()) {
        throw std::runtime_error("cos_theta/phi length mismatch");
    }

    const int64_t npix = x.numel();
    const int64_t nmaps = rows.size(0);
    const int64_t nalm = rows.size(1);
    const int64_t expected_nalm = (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) / 2;
    if (nalm != expected_nalm) {
        throw std::runtime_error("alm length does not match lmax/mmax");
    }

    std::vector<double> ylm_norms(static_cast<size_t>(nalm), 0.0);
    int64_t idx = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        for (int64_t ell = m; ell <= lmax; ++ell) {
            const double log_ratio = std::lgamma(static_cast<double>(ell - m + 1))
                                     - std::lgamma(static_cast<double>(ell + m + 1));
            const double pref = (2.0 * static_cast<double>(ell) + 1.0) / (4.0 * PI);
            ylm_norms[static_cast<size_t>(idx)] = std::sqrt(pref * std::exp(log_ratio));
            ++idx;
        }
    }

    auto out = torch::zeros({nmaps, npix}, torch::TensorOptions().dtype(torch::kDouble));
    const auto* alm_ptr = rows.data_ptr<c10::complex<double>>();
    const auto* x_ptr = x.data_ptr<double>();
    const auto* phi_ptr = ph.data_ptr<double>();
    auto* out_ptr = out.data_ptr<double>();

    at::parallel_for(0, npix, 128, [&](int64_t begin, int64_t end) {
        std::vector<double> acc(static_cast<size_t>(nmaps), 0.0);
        for (int64_t p = begin; p < end; ++p) {
            std::fill(acc.begin(), acc.end(), 0.0);

            const double xp = std::max(-1.0, std::min(1.0, x_ptr[p]));
            const double phi_p = phi_ptr[p];
            const double cphi = std::cos(phi_p);
            const double sphi = std::sin(phi_p);
            double cos_m = 1.0;
            double sin_m = 0.0;
            const double somx2 = std::sqrt(std::max(0.0, 1.0 - xp * xp));
            double pmm = 1.0;
            int64_t k = 0;

            for (int64_t m = 0; m <= mmax; ++m) {
                if (m > 0) {
                    const double cos_next = cos_m * cphi - sin_m * sphi;
                    const double sin_next = sin_m * cphi + cos_m * sphi;
                    cos_m = cos_next;
                    sin_m = sin_next;
                    pmm *= -static_cast<double>(2 * m - 1) * somx2;
                }

                double p_lm = pmm;
                double p_lm_prev = 0.0;
                const double m_weight = (m == 0) ? 1.0 : 2.0;
                for (int64_t ell = m; ell <= lmax; ++ell) {
                    const double y_scale = ylm_norms[static_cast<size_t>(k)] * p_lm;
                    const double y_re = y_scale * cos_m;
                    const double y_im = y_scale * sin_m;

                    for (int64_t r = 0; r < nmaps; ++r) {
                        const auto a = alm_ptr[r * nalm + k];
                        const double term = static_cast<double>(a.real()) * y_re
                                            - static_cast<double>(a.imag()) * y_im;
                        acc[static_cast<size_t>(r)] += m_weight * term;
                    }

                    if (ell < lmax) {
                        double p_next;
                        if (ell == m) {
                            p_next = xp * (2.0 * static_cast<double>(m) + 1.0) * pmm;
                        } else {
                            p_next = ((2.0 * static_cast<double>(ell) + 1.0) * xp * p_lm
                                      - static_cast<double>(ell + m) * p_lm_prev)
                                     / static_cast<double>(ell - m + 1);
                        }
                        p_lm_prev = p_lm;
                        p_lm = p_next;
                    }
                    ++k;
                }
            }

            for (int64_t r = 0; r < nmaps; ++r) {
                out_ptr[r * npix + p] = acc[static_cast<size_t>(r)];
            }
        }
    });

    return out;
}

torch::Tensor healpix_ring_fourier_modes_cpu(
    const torch::Tensor& rows,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax
) {
    if (!rows.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_ring_fourier_modes_cpu expects CPU tensors");
    }
    if (rows.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_ring_fourier_modes_cpu expects complex128 rows");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring_fourier_modes_cpu expects int64 starts/lengths");
    }
    if (rows.dim() != 2 || starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("rows must be rank-2; starts/lengths must be rank-1");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("starts/lengths size mismatch");
    }
    if (mmax < 0) {
        throw std::runtime_error("mmax must be non-negative");
    }
    const int64_t nrows = rows.size(0);
    const int64_t npix = rows.size(1);
    const int64_t nrings = starts.numel();
    const int64_t mcount = mmax + 1;
    if (nrings == 0) {
        return torch::zeros({nrows, mcount, 0}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    }

    auto rows_c = rows.contiguous();
    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    auto out = torch::zeros({nrows, mcount, nrings}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();

    int64_t ring = 0;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        if (nph <= 0 || run_pix_start < 0 || (run_pix_start + nph) > npix) {
            throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_cpu");
        }
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings) {
            const int64_t next_start = starts_ptr[ring];
            const int64_t next_nph = lengths_ptr[ring];
            if (next_nph <= 0 || next_start < 0 || (next_start + next_nph) > npix) {
                throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_cpu");
            }
            if (next_nph != nph || next_start != pix_cursor) {
                break;
            }
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;

        auto block = rows_c.narrow(1, run_pix_start, run_ring_count * nph).view({nrows, run_ring_count, nph});
        auto fft_block = at::fft_fft(block, std::nullopt, -1, std::nullopt);

        torch::Tensor fft_sel;
        if (nph > mmax) {
            fft_sel = fft_block.narrow(2, 0, mcount);
        } else {
            const auto& idx = g_ring_alias_idx_cache.plus(nph, mcount);
            fft_sel = fft_block.index_select(2, idx);
        }
        out.narrow(2, run_start_ring, run_ring_count).copy_(fft_sel.permute({0, 2, 1}));
    }

    return out;
}

torch::Tensor healpix_ring_fourier_modes_spin_conj_cpu(
    const torch::Tensor& row,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax
) {
    if (!row.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu expects CPU tensors");
    }
    if (row.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu expects complex128 row");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu expects int64 starts/lengths");
    }
    if (row.dim() != 1 || starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("row must be rank-1; starts/lengths must be rank-1");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("starts/lengths size mismatch");
    }
    if (mmax < 0) {
        throw std::runtime_error("mmax must be non-negative");
    }

    const int64_t npix = row.size(0);
    const int64_t nrings = starts.numel();
    const int64_t mcount = mmax + 1;
    if (nrings == 0) {
        return torch::zeros({2, mcount, 0}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    }
    auto row_c = row.contiguous();
    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();

    auto out = torch::zeros(
        {2, mcount, nrings},
        torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU)
    );
    auto out_plus = out.select(0, 0);
    auto out_minus = out.select(0, 1);

    int64_t ring = 0;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        if (nph <= 0 || run_pix_start < 0 || (run_pix_start + nph) > npix) {
            throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_spin_conj_cpu");
        }
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings) {
            const int64_t next_start = starts_ptr[ring];
            const int64_t next_nph = lengths_ptr[ring];
            if (next_nph <= 0 || next_start < 0 || (next_start + next_nph) > npix) {
                throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_spin_conj_cpu");
            }
            if (next_nph != nph || next_start != pix_cursor) {
                break;
            }
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;

        auto block = row_c.narrow(0, run_pix_start, run_ring_count * nph).view({run_ring_count, nph});
        auto fft_block = at::fft_fft(block, std::nullopt, -1, std::nullopt);

        torch::Tensor plus_sel;
        torch::Tensor minus_sel;
        if (nph > mmax) {
            plus_sel = fft_block.narrow(1, 0, mcount);
            const auto& idx_minus = g_ring_alias_idx_cache.minus(nph, mcount);
            minus_sel = torch::conj(fft_block.index_select(1, idx_minus));
        } else {
            const auto& idx_both = g_ring_alias_idx_cache.both(nph, mcount);
            auto both_sel = fft_block.index_select(1, idx_both);
            plus_sel = both_sel.narrow(1, 0, mcount);
            minus_sel = torch::conj(both_sel.narrow(1, mcount, mcount));
        }

        out_plus.narrow(1, run_start_ring, run_ring_count).copy_(plus_sel.transpose(0, 1));
        out_minus.narrow(1, run_start_ring, run_ring_count).copy_(minus_sel.transpose(0, 1));
    }

    return out;
}

torch::Tensor healpix_ring_fourier_synthesis_cpu(
    const torch::Tensor& s_modes,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
) {
    if (!s_modes.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_ring_fourier_synthesis_cpu expects CPU tensors");
    }
    if (s_modes.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_ring_fourier_synthesis_cpu expects complex128 s_modes");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring_fourier_synthesis_cpu expects int64 starts/lengths");
    }
    if (starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("starts/lengths must be rank-1");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("starts/lengths size mismatch");
    }

    torch::Tensor s_batched;
    bool squeeze_out = false;
    if (s_modes.dim() == 2) {
        s_batched = s_modes.contiguous().unsqueeze(0);
        squeeze_out = true;
    } else if (s_modes.dim() == 3) {
        s_batched = s_modes.contiguous();
    } else {
        throw std::runtime_error("s_modes must be rank-2 or rank-3");
    }

    const int64_t nrows = s_batched.size(0);
    const int64_t mcount = s_batched.size(1);
    const int64_t nrings = s_batched.size(2);
    if (nrings != starts.numel()) {
        throw std::runtime_error("s_modes ring dimension does not match starts/lengths");
    }
    if (mcount <= 0) {
        throw std::runtime_error("s_modes m dimension must be positive");
    }

    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const int64_t npix = starts_ptr[nrings - 1] + lengths_ptr[nrings - 1];
    if (npix < 0) {
        throw std::runtime_error("invalid npix in healpix_ring_fourier_synthesis_cpu");
    }

    auto out = torch::zeros({nrows, npix}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    const auto cplx_opts = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    const int64_t mmax = mcount - 1;

    int64_t ring = 0;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        if (nph <= 0 || run_pix_start < 0 || (run_pix_start + nph) > npix) {
            throw std::runtime_error("invalid ring layout in healpix_ring_fourier_synthesis_cpu");
        }
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings) {
            const int64_t next_start = starts_ptr[ring];
            const int64_t next_nph = lengths_ptr[ring];
            if (next_nph <= 0 || next_start < 0 || (next_start + next_nph) > npix) {
                throw std::runtime_error("invalid ring layout in healpix_ring_fourier_synthesis_cpu");
            }
            if (next_nph != nph || next_start != pix_cursor) {
                break;
            }
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;

        auto s_group = s_batched.narrow(2, run_start_ring, run_ring_count).permute({0, 2, 1});
        torch::Tensor map_vals;
        if (nph > mmax) {
            map_vals = at::fft_ifft(s_group * static_cast<double>(nph), nph, -1, std::nullopt);
        } else {
            auto s_fold = torch::zeros({nrows, run_ring_count, nph}, cplx_opts);
            const auto& idx = g_ring_alias_idx_cache.plus(nph, mcount);
            s_fold.index_add_(2, idx, s_group);
            map_vals = at::fft_ifft(s_fold * static_cast<double>(nph), std::nullopt, -1, std::nullopt);
        }

        out.narrow(1, run_pix_start, run_ring_count * nph).copy_(map_vals.reshape({nrows, run_ring_count * nph}));
    }

    if (squeeze_out) {
        return out.reshape({npix});
    }
    return out;
}

torch::Tensor healpix_spin_ring_finalize_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
) {
    if (!s_plus.device().is_cpu() || !s_minus.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects CPU tensors");
    }
    if (s_plus.scalar_type() != torch::kComplexDouble || s_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects complex128 s_plus/s_minus");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects int64 starts/lengths");
    }
    if (s_plus.dim() != 2 || s_minus.dim() != 2 || starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects rank-2 s tensors and rank-1 starts/lengths");
    }
    if (s_plus.sizes() != s_minus.sizes()) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu s_plus/s_minus shape mismatch");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu starts/lengths size mismatch");
    }
    const int64_t nrings = starts.numel();
    if (s_plus.size(1) != nrings) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu ring dimension mismatch");
    }

    auto sp = s_plus.contiguous();
    auto sm = s_minus.contiguous();
    auto s_stacked = torch::stack({sp, sm}, 0);
    auto p_pm_pos = healpix_ring_fourier_synthesis_cpu(s_stacked, starts, lengths);
    auto p_plus_pos = p_pm_pos.select(0, 0).contiguous();
    auto p_minus_pos = p_pm_pos.select(0, 1).contiguous();

    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const int64_t npix = p_plus_pos.size(0);
    const int64_t mmax = sp.size(0) - 1;
    auto sp_a = sp.accessor<c10::complex<double>, 2>();
    auto sm_a = sm.accessor<c10::complex<double>, 2>();

    // Small-map path: keep the old tensor expression path, which can be faster for tiny workloads.
    if (npix < 4096) {
        auto p_plus_m0 = torch::empty({npix}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
        auto p_minus_m0 = torch::empty_like(p_plus_m0);
        auto* p_plus_m0_ptr = p_plus_m0.data_ptr<c10::complex<double>>();
        auto* p_minus_m0_ptr = p_minus_m0.data_ptr<c10::complex<double>>();
        for (int64_t r = 0; r < nrings; ++r) {
            const int64_t start = starts_ptr[r];
            const int64_t nph = lengths_ptr[r];
            if (nph <= 0 || start < 0 || start + nph > npix) {
                throw std::runtime_error("invalid ring layout in healpix_spin_ring_finalize_cpu");
            }
            const c10::complex<double> v_plus = sp_a[0][r];
            const c10::complex<double> v_minus = sm_a[0][r];
            for (int64_t i = 0; i < nph; ++i) {
                p_plus_m0_ptr[start + i] = v_plus;
                p_minus_m0_ptr[start + i] = v_minus;
            }
        }

        torch::Tensor p_plus;
        torch::Tensor p_minus;
        if (mmax == 0) {
            p_plus = p_plus_pos;
            p_minus = p_minus_pos;
        } else {
            p_plus = p_plus_pos + torch::conj(p_minus_pos - p_minus_m0);
            p_minus = p_minus_pos + torch::conj(p_plus_pos - p_plus_m0);
        }
        auto q = -0.5 * torch::real(p_plus + p_minus);
        auto u = -0.5 * torch::imag(p_plus - p_minus);
        return torch::stack({q.to(torch::kFloat64), u.to(torch::kFloat64)}, 0);
    }

    auto out = torch::empty({2, npix}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    const auto* p_plus_ptr = p_plus_pos.data_ptr<c10::complex<double>>();
    const auto* p_minus_ptr = p_minus_pos.data_ptr<c10::complex<double>>();
    auto* q_ptr = out.select(0, 0).data_ptr<double>();
    auto* u_ptr = out.select(0, 1).data_ptr<double>();

    for (int64_t r = 0; r < nrings; ++r) {
        const int64_t start = starts_ptr[r];
        const int64_t nph = lengths_ptr[r];
        if (nph <= 0 || start < 0 || start + nph > npix) {
            throw std::runtime_error("invalid ring layout in healpix_spin_ring_finalize_cpu");
        }
        const c10::complex<double> v_plus_m0 = sp_a[0][r];
        const c10::complex<double> v_minus_m0 = sm_a[0][r];
        const c10::complex<double> v_plus_m0_conj(v_plus_m0.real(), -v_plus_m0.imag());
        const c10::complex<double> v_minus_m0_conj(v_minus_m0.real(), -v_minus_m0.imag());

        for (int64_t i = 0; i < nph; ++i) {
            const int64_t p = start + i;
            const c10::complex<double> a = p_plus_ptr[p];
            const c10::complex<double> b = p_minus_ptr[p];
            c10::complex<double> p_plus;
            c10::complex<double> p_minus;
            if (mmax == 0) {
                p_plus = a;
                p_minus = b;
            } else {
                const c10::complex<double> b_conj(b.real(), -b.imag());
                const c10::complex<double> a_conj(a.real(), -a.imag());
                p_plus = a + (b_conj - v_minus_m0_conj);
                p_minus = b + (a_conj - v_plus_m0_conj);
            }
            const c10::complex<double> sum = p_plus + p_minus;
            const c10::complex<double> diff = p_plus - p_minus;
            q_ptr[p] = -0.5 * sum.real();
            u_ptr[p] = -0.5 * diff.imag();
        }
    }
    return out;
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_map2alm_from_basis_cpu(
    const torch::Tensor& q,
    const torch::Tensor& u,
    const torch::Tensor& ycp_t,
    const torch::Tensor& ycm_t,
    double pix_w
) {
    if (!q.device().is_cpu() || !u.device().is_cpu() || !ycp_t.device().is_cpu() || !ycm_t.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_map2alm_from_basis_cpu expects CPU tensors");
    }
    if (q.scalar_type() != torch::kDouble || u.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_spin_map2alm_from_basis_cpu expects float64 q/u tensors");
    }
    if (ycp_t.scalar_type() != torch::kComplexDouble || ycm_t.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_map2alm_from_basis_cpu expects complex128 basis tensors");
    }

    auto qv = q.contiguous().reshape({-1});
    auto uv = u.contiguous().reshape({-1});
    auto ycp = ycp_t.contiguous();
    auto ycm = ycm_t.contiguous();

    if (qv.numel() != uv.numel()) {
        throw std::runtime_error("q/u length mismatch");
    }
    if (ycp.dim() != 2 || ycm.dim() != 2) {
        throw std::runtime_error("basis tensors must be rank-2");
    }
    if (ycp.sizes() != ycm.sizes()) {
        throw std::runtime_error("basis tensors shape mismatch");
    }

    const int64_t npix = qv.numel();
    if (ycp.size(0) != npix) {
        throw std::runtime_error("basis first dimension must match q/u length");
    }
    auto p_plus = torch::complex(qv, uv);
    auto p_minus = torch::conj(p_plus);
    auto c_plus = torch::matmul(p_plus, ycp) * pix_w;
    auto c_minus = torch::matmul(p_minus, ycm) * pix_w;
    return {c_plus, c_minus};
}

torch::Tensor healpix_spin_alm2map_from_basis_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    const torch::Tensor& y_plus_m0,
    const torch::Tensor& y_minus_m0
) {
    if (
        !coeff_plus.device().is_cpu() || !coeff_minus.device().is_cpu() || !y_plus.device().is_cpu()
        || !y_minus.device().is_cpu() || !y_plus_m0.device().is_cpu() || !y_minus_m0.device().is_cpu()
    ) {
        throw std::runtime_error("healpix_spin_alm2map_from_basis_cpu expects CPU tensors");
    }
    if (
        coeff_plus.scalar_type() != torch::kComplexDouble || coeff_minus.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble
        || y_plus_m0.scalar_type() != torch::kComplexDouble || y_minus_m0.scalar_type() != torch::kComplexDouble
    ) {
        throw std::runtime_error("healpix_spin_alm2map_from_basis_cpu expects complex128 tensors");
    }

    auto cp = coeff_plus.contiguous().reshape({-1});
    auto cm = coeff_minus.contiguous().reshape({-1});
    auto yp = y_plus.contiguous();
    auto ym = y_minus.contiguous();
    auto yp0 = y_plus_m0.contiguous();
    auto ym0 = y_minus_m0.contiguous();

    if (cp.numel() != cm.numel()) {
        throw std::runtime_error("coeff_plus/coeff_minus length mismatch");
    }
    if (yp.dim() != 2 || ym.dim() != 2 || yp0.dim() != 2 || ym0.dim() != 2) {
        throw std::runtime_error("basis tensors must be rank-2");
    }
    if (yp.sizes() != ym.sizes()) {
        throw std::runtime_error("y_plus/y_minus shape mismatch");
    }
    if (yp0.sizes() != ym0.sizes()) {
        throw std::runtime_error("y_plus_m0/y_minus_m0 shape mismatch");
    }
    const int64_t nalm = cp.numel();
    if (yp.size(0) != nalm) {
        throw std::runtime_error("basis row count must match coefficient length");
    }
    const int64_t npix = yp.size(1);
    if (ym.size(1) != npix || yp0.size(1) != npix || ym0.size(1) != npix) {
        throw std::runtime_error("basis tensors must have the same pixel dimension");
    }
    const int64_t l0 = yp0.size(0);
    if (l0 > nalm) {
        throw std::runtime_error("m0 basis row count cannot exceed coefficient length");
    }

    auto p_plus_pos = torch::matmul(cp, yp);
    auto p_minus_pos = torch::matmul(cm, ym);
    torch::Tensor p_plus;
    torch::Tensor p_minus;
    if (l0 == nalm) {
        p_plus = p_plus_pos;
        p_minus = p_minus_pos;
    } else {
        auto cp0 = cp.narrow(0, 0, l0);
        auto cm0 = cm.narrow(0, 0, l0);
        auto p_plus_m0 = torch::matmul(cp0, yp0);
        auto p_minus_m0 = torch::matmul(cm0, ym0);
        p_plus = p_plus_pos + torch::conj(p_minus_pos - p_minus_m0);
        p_minus = p_minus_pos + torch::conj(p_plus_pos - p_plus_m0);
    }

    auto q = -0.5 * torch::real(p_plus + p_minus);
    auto u = -0.5 * torch::imag(p_plus - p_minus);
    return torch::stack({q.to(torch::kFloat64), u.to(torch::kFloat64)}, 0);
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_interpolate_concat_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax
) {
    if (!coeff_plus.device().is_cpu() || !coeff_minus.device().is_cpu() || !y_plus.device().is_cpu()
        || !y_minus.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu expects CPU tensors");
    }
    if (coeff_plus.scalar_type() != torch::kComplexDouble || coeff_minus.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu expects complex128 tensors");
    }
    if (coeff_plus.dim() != 1 || coeff_minus.dim() != 1 || y_plus.dim() != 2 || y_minus.dim() != 2) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu expects coeff rank-1 and basis rank-2");
    }
    if (coeff_plus.numel() != coeff_minus.numel()) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu coeff length mismatch");
    }
    if (y_plus.sizes() != y_minus.sizes()) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu basis shape mismatch");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu invalid lmax/mmax");
    }

    auto cp = coeff_plus.contiguous();
    auto cm = coeff_minus.contiguous();
    auto yp = y_plus.contiguous();
    auto ym = y_minus.contiguous();

    const int64_t nalm = cp.numel();
    const int64_t nrings = yp.size(1);
    const int64_t mcount = mmax + 1;
    if (yp.size(0) != nalm) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu basis row count must match coeff length");
    }
    if (alm_size(lmax, mmax) != nalm) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu nalm does not match lmax/mmax");
    }

    auto out_plus = torch::zeros({mcount, nrings}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto out_minus = torch::zeros_like(out_plus);

    const auto* cp_ptr = cp.data_ptr<c10::complex<double>>();
    const auto* cm_ptr = cm.data_ptr<c10::complex<double>>();
    const auto* yp_ptr = yp.data_ptr<c10::complex<double>>();
    const auto* ym_ptr = ym.data_ptr<c10::complex<double>>();
    auto* op_ptr = out_plus.data_ptr<c10::complex<double>>();
    auto* om_ptr = out_minus.data_ptr<c10::complex<double>>();

    at::parallel_for(0, mcount, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const int64_t base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const int64_t l_count = lmax - m + 1;
            auto* op_row = op_ptr + m * nrings;
            auto* om_row = om_ptr + m * nrings;
            for (int64_t r = 0; r < nrings; ++r) {
                c10::complex<double> accp(0.0, 0.0);
                c10::complex<double> accm(0.0, 0.0);
                for (int64_t t = 0; t < l_count; ++t) {
                    const int64_t i = base + t;
                    accp += cp_ptr[i] * yp_ptr[i * nrings + r];
                    accm += cm_ptr[i] * ym_ptr[i * nrings + r];
                }
                op_row[r] = accp;
                om_row[r] = accm;
            }
        }
    });

    return {out_plus, out_minus};
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_integrate_concat_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
) {
    if (!s_plus.device().is_cpu() || !s_minus.device().is_cpu() || !y_plus.device().is_cpu()
        || !y_minus.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu expects CPU tensors");
    }
    if (s_plus.scalar_type() != torch::kComplexDouble || s_minus.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu expects complex128 tensors");
    }
    if (s_plus.dim() != 2 || s_minus.dim() != 2 || y_plus.dim() != 2 || y_minus.dim() != 2) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu expects rank-2 tensors");
    }
    if (s_plus.sizes() != s_minus.sizes()) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu s_plus/s_minus shape mismatch");
    }
    if (y_plus.sizes() != y_minus.sizes()) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu y_plus/y_minus shape mismatch");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu invalid lmax/mmax");
    }

    auto sp = s_plus.contiguous();
    auto sm = s_minus.contiguous();
    auto yp = y_plus.contiguous();
    auto ym = y_minus.contiguous();

    const int64_t mcount = mmax + 1;
    const int64_t nrings = sp.size(1);
    const int64_t nalm = yp.size(0);
    if (sp.size(0) != mcount || sm.size(0) != mcount) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu s_plus/s_minus first dim must be mmax+1");
    }
    if (yp.size(1) != nrings || ym.size(1) != nrings) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu basis ring dim mismatch");
    }
    if (alm_size(lmax, mmax) != nalm) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu nalm does not match lmax/mmax");
    }

    auto out_plus = torch::zeros({nalm}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto out_minus = torch::zeros_like(out_plus);

    const auto* sp_ptr = sp.data_ptr<c10::complex<double>>();
    const auto* sm_ptr = sm.data_ptr<c10::complex<double>>();
    const auto* yp_ptr = yp.data_ptr<c10::complex<double>>();
    const auto* ym_ptr = ym.data_ptr<c10::complex<double>>();
    auto* op_ptr = out_plus.data_ptr<c10::complex<double>>();
    auto* om_ptr = out_minus.data_ptr<c10::complex<double>>();

    at::parallel_for(0, mcount, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const int64_t base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const int64_t l_count = lmax - m + 1;
            const auto* sp_row = sp_ptr + m * nrings;
            const auto* sm_row = sm_ptr + m * nrings;
            for (int64_t t = 0; t < l_count; ++t) {
                const int64_t i = base + t;
                const auto* yp_row = yp_ptr + i * nrings;
                const auto* ym_row = ym_ptr + i * nrings;
                c10::complex<double> accp(0.0, 0.0);
                c10::complex<double> accm(0.0, 0.0);
                for (int64_t r = 0; r < nrings; ++r) {
                    accp += yp_row[r] * sp_row[r];
                    accm += ym_row[r] * sm_row[r];
                }
                op_ptr[i] = accp * pix_w;
                om_ptr[i] = accm * pix_w;
            }
        }
    });

    return {out_plus, out_minus};
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_map2alm_ring_concat_cpu(
    const torch::Tensor& p_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& phase0_neg,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
) {
    if (!p_plus.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()
        || !phase0_neg.device().is_cpu() || !y_plus.device().is_cpu() || !y_minus.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu expects CPU tensors");
    }
    if (p_plus.scalar_type() != torch::kComplexDouble || phase0_neg.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu expects complex128 tensors");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu expects int64 starts/lengths");
    }
    if (p_plus.dim() != 1 || starts.dim() != 1 || lengths.dim() != 1 || phase0_neg.dim() != 2
        || y_plus.dim() != 2 || y_minus.dim() != 2) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu invalid tensor ranks");
    }
    if (y_plus.sizes() != y_minus.sizes()) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu y_plus/y_minus shape mismatch");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu invalid lmax/mmax");
    }
    const int64_t mcount = mmax + 1;
    auto s_stacked = healpix_ring_fourier_modes_spin_conj_cpu(p_plus, starts, lengths, mmax);
    auto phase = phase0_neg.contiguous();
    auto s_plus = s_stacked.select(0, 0);
    auto s_minus = s_stacked.select(0, 1);
    if (phase.size(0) != mcount || phase.size(1) != s_plus.size(1)) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu phase0_neg shape mismatch");
    }
    s_plus = s_plus * phase;
    s_minus = s_minus * phase;
    return healpix_spin_integrate_concat_cpu(s_plus, s_minus, y_plus, y_minus, lmax, mmax, pix_w);
}

std::pair<torch::Tensor, torch::Tensor> healpix_get_interp_weights_ring_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!lon_deg.device().is_cpu() || !lat_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_get_interp_weights_ring_cpu expects CPU tensors");
    }
    if (lon_deg.scalar_type() != torch::kDouble || lat_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_get_interp_weights_ring_cpu expects float64 tensors");
    }
    if (!lon_deg.sizes().equals(lat_deg.sizes())) {
        throw std::runtime_error("lon/lat shape mismatch");
    }

    auto lon = lon_deg.contiguous().reshape({-1});
    auto lat = lat_deg.contiguous().reshape({-1});
    const int64_t n = lon.numel();
    auto pix = torch::empty({4, n}, torch::TensorOptions().dtype(torch::kInt64));
    auto wgt = torch::zeros({4, n}, torch::TensorOptions().dtype(torch::kDouble));

    const auto* lon_ptr = lon.data_ptr<double>();
    const auto* lat_ptr = lat.data_ptr<double>();
    auto* p_ptr = pix.data_ptr<int64_t>();
    auto* w_ptr = wgt.data_ptr<double>();

    const int64_t npix = 12 * nside * nside;
    const int64_t n4 = 4 * nside;

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const double theta = (90.0 - lat_ptr[i]) * PI / 180.0;
            const double phi = wrap_rad_2pi(lon_ptr[i] * PI / 180.0);
            const double z = std::cos(theta);
            const int64_t ir1 = ring_above_scalar(nside, z);
            const int64_t ir2 = ir1 + 1;

            int64_t p0 = -1;
            int64_t p1 = -1;
            int64_t p2 = -1;
            int64_t p3 = -1;
            double w0 = 0.0;
            double w1 = 0.0;
            double w2 = 0.0;
            double w3 = 0.0;
            double theta1 = 0.0;
            double theta2 = 0.0;

            if (ir1 > 0) {
                int64_t sp;
                int64_t nr;
                bool shifted;
                get_ring_info2_scalar(nside, ir1, sp, nr, theta1, shifted);
                const double sh = shifted ? 1.0 : 0.0;
                const double dphi = TWO_PI / static_cast<double>(nr);
                const double tmp = phi / dphi - 0.5 * sh;
                int64_t i1 = static_cast<int64_t>(std::floor(tmp));
                const double wt = (phi - (static_cast<double>(i1) + 0.5 * sh) * dphi) / dphi;
                int64_t i2 = i1 + 1;
                if (i1 < 0) {
                    i1 += nr;
                }
                if (i2 >= nr) {
                    i2 -= nr;
                }
                p0 = sp + i1;
                p1 = sp + i2;
                w0 = 1.0 - wt;
                w1 = wt;
            }

            if (ir2 < n4) {
                int64_t sp;
                int64_t nr;
                bool shifted;
                get_ring_info2_scalar(nside, ir2, sp, nr, theta2, shifted);
                const double sh = shifted ? 1.0 : 0.0;
                const double dphi = TWO_PI / static_cast<double>(nr);
                const double tmp = phi / dphi - 0.5 * sh;
                int64_t i1 = static_cast<int64_t>(std::floor(tmp));
                const double wt = (phi - (static_cast<double>(i1) + 0.5 * sh) * dphi) / dphi;
                int64_t i2 = i1 + 1;
                if (i1 < 0) {
                    i1 += nr;
                }
                if (i2 >= nr) {
                    i2 -= nr;
                }
                p2 = sp + i1;
                p3 = sp + i2;
                w2 = 1.0 - wt;
                w3 = wt;
            }

            if (ir1 == 0) {
                const double wtheta = theta / theta2;
                w2 *= wtheta;
                w3 *= wtheta;
                const double fac = (1.0 - wtheta) * 0.25;
                w0 = fac;
                w1 = fac;
                w2 += fac;
                w3 += fac;
                p0 = (p2 + 2) & 3;
                p1 = (p3 + 2) & 3;
            } else if (ir2 == n4) {
                const double wtheta = (theta - theta1) / (PI - theta1);
                w0 *= (1.0 - wtheta);
                w1 *= (1.0 - wtheta);
                const double fac = wtheta * 0.25;
                w0 += fac;
                w1 += fac;
                w2 = fac;
                w3 = fac;
                p2 = ((p0 + 2) & 3) + npix - 4;
                p3 = ((p1 + 2) & 3) + npix - 4;
            } else {
                const double wtheta = (theta - theta1) / (theta2 - theta1);
                w0 *= (1.0 - wtheta);
                w1 *= (1.0 - wtheta);
                w2 *= wtheta;
                w3 *= wtheta;
            }

            p_ptr[i] = p0;
            p_ptr[n + i] = p1;
            p_ptr[2 * n + i] = p2;
            p_ptr[3 * n + i] = p3;
            w_ptr[i] = w0;
            w_ptr[n + i] = w1;
            w_ptr[2 * n + i] = w2;
            w_ptr[3 * n + i] = w3;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 4096, compute_block);
    } else {
        compute_block(0, n);
    }

    return {pix, wgt};
}

std::pair<torch::Tensor, torch::Tensor> healpix_get_interp_weights_nested_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    auto out = healpix_get_interp_weights_ring_cpu(nside, lon_deg, lat_deg);
    auto pix_n = healpix_ring2nest_cpu(nside, out.first.reshape({-1})).reshape_as(out.first);
    return {pix_n, out.second};
}

static torch::Tensor healpix_interp_val_from_weights_cpu(
    const torch::Tensor& maps_2d,
    const torch::Tensor& pix4n,
    const torch::Tensor& wgt4n
) {
    auto pix_flat = pix4n.contiguous().reshape({-1});
    const int64_t nq = pix4n.size(1);
    auto gathered = maps_2d.index_select(1, pix_flat).reshape({maps_2d.size(0), 4, nq});
    auto w = wgt4n.to(maps_2d.scalar_type()).unsqueeze(0);
    return (gathered * w).sum(1);
}

torch::Tensor healpix_get_interp_val_ring_cpu(
    int64_t nside,
    const torch::Tensor& maps_in,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    if (!maps_in.device().is_cpu() || !lon_deg.device().is_cpu() || !lat_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_get_interp_val_ring_cpu expects CPU tensors");
    }
    if (lon_deg.scalar_type() != torch::kDouble || lat_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_get_interp_val_ring_cpu expects float64 lon/lat tensors");
    }
    if (!lon_deg.sizes().equals(lat_deg.sizes())) {
        throw std::runtime_error("lon/lat shape mismatch");
    }
    if (maps_in.dim() != 1 && maps_in.dim() != 2) {
        throw std::runtime_error("maps must be rank-1 or rank-2");
    }
    if (
        maps_in.scalar_type() != torch::kFloat && maps_in.scalar_type() != torch::kDouble
        && maps_in.scalar_type() != torch::kBFloat16 && maps_in.scalar_type() != torch::kHalf
    ) {
        throw std::runtime_error("maps must be floating point");
    }

    const bool one_map = maps_in.dim() == 1;
    auto maps = one_map ? maps_in.contiguous().reshape({1, -1}) : maps_in.contiguous();
    const int64_t npix = maps.size(1);
    if (npix != 12 * nside * nside) {
        throw std::runtime_error("maps second dimension must be 12*nside*nside");
    }

    auto out_w = healpix_get_interp_weights_ring_cpu(nside, lon_deg, lat_deg);
    auto values = healpix_interp_val_from_weights_cpu(maps, out_w.first, out_w.second);

    std::vector<int64_t> qshape = lon_deg.sizes().vec();
    if (one_map) {
        return values.reshape(qshape);
    }
    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(qshape.size()) + 1);
    out_shape.push_back(maps.size(0));
    out_shape.insert(out_shape.end(), qshape.begin(), qshape.end());
    return values.reshape(out_shape);
}

torch::Tensor healpix_get_interp_val_nested_cpu(
    int64_t nside,
    const torch::Tensor& maps_in,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    if (!maps_in.device().is_cpu() || !lon_deg.device().is_cpu() || !lat_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_get_interp_val_nested_cpu expects CPU tensors");
    }
    if (lon_deg.scalar_type() != torch::kDouble || lat_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_get_interp_val_nested_cpu expects float64 lon/lat tensors");
    }
    if (!lon_deg.sizes().equals(lat_deg.sizes())) {
        throw std::runtime_error("lon/lat shape mismatch");
    }
    if (maps_in.dim() != 1 && maps_in.dim() != 2) {
        throw std::runtime_error("maps must be rank-1 or rank-2");
    }
    if (
        maps_in.scalar_type() != torch::kFloat && maps_in.scalar_type() != torch::kDouble
        && maps_in.scalar_type() != torch::kBFloat16 && maps_in.scalar_type() != torch::kHalf
    ) {
        throw std::runtime_error("maps must be floating point");
    }

    const bool one_map = maps_in.dim() == 1;
    auto maps = one_map ? maps_in.contiguous().reshape({1, -1}) : maps_in.contiguous();
    const int64_t npix = maps.size(1);
    if (npix != 12 * nside * nside) {
        throw std::runtime_error("maps second dimension must be 12*nside*nside");
    }

    auto out_w = healpix_get_interp_weights_nested_cpu(nside, lon_deg, lat_deg);
    auto values = healpix_interp_val_from_weights_cpu(maps, out_w.first, out_w.second);

    std::vector<int64_t> qshape = lon_deg.sizes().vec();
    if (one_map) {
        return values.reshape(qshape);
    }
    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(qshape.size()) + 1);
    out_shape.push_back(maps.size(0));
    out_shape.insert(out_shape.end(), qshape.begin(), qshape.end());
    return values.reshape(out_shape);
}

// Helper for Wigner-d element (l near m case)
static double wigner_d_element_scalar(
    int64_t l,
    int64_t m,
    int64_t mp,
    double theta
) {
    // Matches Python _wigner_d_element implementation
    // d^l_{m,mp}(theta)

    double log_pref = 0.5 * (std::lgamma(l + m + 1) + std::lgamma(l - m + 1) + 
                             std::lgamma(l + mp + 1) + std::lgamma(l - mp + 1));
                             
    // Python bounds:
    // kmin = max(0, m - mp)
    // kmax = min(l + m, l - mp)
    int64_t k_min = std::max((int64_t)0, m - mp);
    int64_t k_max = std::min(l + m, l - mp);
    
    if (k_min > k_max) return 0.0;
    
    double sum = 0.0;
    double sin_half = std::sin(theta * 0.5);
    double cos_half = std::cos(theta * 0.5);
    
    double log_sin = (sin_half > 0) ? std::log(sin_half) : -1.0e10; // Handle 0 safely if needed
    double log_cos = (cos_half > 0) ? std::log(cos_half) : -1.0e10;
    
    // For 0 or PI, special handling might be preferred, but let's stick to formula with safeguards
    // If sin_half=0 (theta=0), we only want terms with 0 power for sin.
    // If cos_half=0 (theta=pi), we only want terms with 0 power for cos.
    
    for(int64_t k = k_min; k <= k_max; ++k) {
        int64_t a = l + m - k;
        int64_t b = k;
        int64_t c = mp - m + k;
        int64_t d = l - mp - k;
        
        // This check is redundant with loop bounds but safe
        if (a < 0 || b < 0 || c < 0 || d < 0) continue;
        
        double log_denom = std::lgamma(a + 1) + std::lgamma(b + 1) + 
                           std::lgamma(c + 1) + std::lgamma(d + 1);
        
        double sign = ((k + mp - m) % 2 != 0) ? -1.0 : 1.0;
        
        double p_ct = 2 * l + m - mp - 2 * k;
        double p_st = mp - m + 2 * k;
        
        // Safe power logic
        double term_val = 0.0;
        bool zero_term = false;
        
        // Handle cos power
        if (cos_half <= 0.0) {
             if (p_ct == 0) term_val += 0.0; // log(1)
             else zero_term = true;
        } else {
             term_val += p_ct * log_cos;
        }
        
        // Handle sin power
        if (sin_half <= 0.0) {
             if (p_st == 0) term_val += 0.0; // log(1)
             else zero_term = true;
        } else {
             term_val += p_st * log_sin;
        }
        
        if (zero_term) continue;
        
        term_val += log_pref - log_denom;
        sum += sign * std::exp(term_val);
    }
    return sum;
}














static std::vector<double> precomputed_l_factors;
static std::mutex l_factors_mutex;

void ensure_l_factors(int64_t lmax) {
    std::lock_guard<std::mutex> lock(l_factors_mutex);
    if (precomputed_l_factors.size() <= (size_t)lmax) {
        precomputed_l_factors.resize(lmax + 1);
        for (int64_t l = 0; l <= lmax; ++l) {
            precomputed_l_factors[l] = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI));
        }
    }
}

torch::Tensor healpix_spin_integrate_recurrence_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& theta,
    const torch::Tensor& weight,
    int64_t lmax,
    int64_t mmax,
    int64_t spin
) {
    auto sp_f = s_plus.contiguous(); auto sm_f = s_minus.contiguous();
    auto sp_a = sp_f.accessor<c10::complex<double>, 2>();
    auto sm_a = sm_f.accessor<c10::complex<double>, 2>();
    auto th_ptr = theta.data_ptr<double>();
    auto w_ptr = weight.data_ptr<double>();
    
    int64_t nrings = theta.size(0);
    int64_t n_eff = (nrings + 1) / 2;
    int64_t n_pairs = nrings / 2;
    bool has_equator = (nrings % 2 != 0);
    ensure_l_factors(lmax);
    
    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor alms_plus = torch::zeros({alm_size(lmax, mmax)}, options);
    torch::Tensor alms_minus = torch::zeros({alm_size(lmax, mmax)}, options);
    auto cp_out = alms_plus.data_ptr<c10::complex<double>>();
    auto cm_out = alms_minus.data_ptr<c10::complex<double>>();

    std::vector<int64_t> m_offsets(mmax + 1);
    int64_t current_lut_off = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        m_offsets[m] = current_lut_off;
        int64_t l0 = std::max((int64_t)std::abs(spin), m);
        if (lmax >= l0 + 2) {
            current_lut_off += (lmax - (l0 + 2) + 1);
        }
    }
    const auto& lut = g_recurrence_cache.get_coeffs(lmax, mmax, spin, m_offsets);

    std::vector<double> cos_theta_all(n_eff);
    for(int i=0; i<n_eff; ++i) cos_theta_all[i] = std::cos(th_ptr[i]);
    std::vector<int64_t> mlim_north(n_eff, lmax);
    if (spin_mlim_prune_enabled()) {
        const int64_t spin_abs = std::abs(spin);
        for (int64_t i = 0; i < n_eff; ++i) {
            const double cth = cos_theta_all[static_cast<size_t>(i)];
            const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
            mlim_north[static_cast<size_t>(i)] = sharp_mlim_estimate(lmax, spin_abs, sth, cth);
        }
    }

    // For small N, overhead of parallel_for is high. Use a larger grain size (e.g. 8).
    at::parallel_for(0, mmax + 1, 8, [&](int64_t m_start, int64_t m_end) {
        // Allocate working buffers ONCE per thread task, rather than per inner loop.
        std::vector<double> v_dp2(n_eff), v_dp1(n_eff), v_dpc(n_eff);
        std::vector<double> v_dm2(n_eff), v_dm1(n_eff), v_dmc(n_eff);
        std::vector<uint8_t> v_active(n_eff);
        double * __restrict__ dp2 = v_dp2.data();
        double * __restrict__ dp1 = v_dp1.data();
        double * __restrict__ dpc = v_dpc.data();
        double * __restrict__ dm2 = v_dm2.data();
        double * __restrict__ dm1 = v_dm1.data();
        double * __restrict__ dmc = v_dmc.data();
        uint8_t * __restrict__ active = v_active.data();
        const double * __restrict__ ct_ptr = cos_theta_all.data();

        std::vector<double> vNp_r(n_eff), vNp_i(n_eff), vNm_r(n_eff), vNm_i(n_eff);
        std::vector<double> vSp_r(n_eff), vSp_i(n_eff), vSm_r(n_eff), vSm_i(n_eff);
        std::vector<double> v_weights(n_eff), v_weightsS(n_eff);
        double * __restrict__ Np_r = vNp_r.data(); double * __restrict__ Np_i = vNp_i.data();
        double * __restrict__ Nm_r = vNm_r.data(); double * __restrict__ Nm_i = vNm_i.data();
        double * __restrict__ Sp_r = vSp_r.data(); double * __restrict__ Sp_i = vSp_i.data();
        double * __restrict__ Sm_r = vSm_r.data(); double * __restrict__ Sm_i = vSm_i.data();
        double * __restrict__ weightsN = v_weights.data();
        double * __restrict__ weightsS = v_weightsS.data();

        for (int64_t m = m_start; m < m_end; ++m) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            int64_t idx_base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const double pre_phase = ((m % 2 != 0) ? -1.0 : 1.0) * ((std::abs(spin) % 2 != 0) ? -1.0 : 1.0);
            const bool spin_active = (spin != 0);
            const auto* lut_ptr = &lut[m_offsets[m]];
            bool any_active = false;

            for (int i=0; i < n_eff; ++i) {
                const bool on = (m <= mlim_north[static_cast<size_t>(i)]);
                active[i] = static_cast<uint8_t>(on ? 1 : 0);
                any_active = any_active || on;
                if (!on) {
                    continue;
                }
                weightsN[i] = w_ptr[i];
                Np_r[i] = sp_a[m][i].real(); Np_i[i] = sp_a[m][i].imag();
                Nm_r[i] = sm_a[m][i].real(); Nm_i[i] = sm_a[m][i].imag();
                if (i < n_pairs) {
                    int64_t idx_S = nrings - 1 - i;
                    weightsS[i] = w_ptr[idx_S];
                    Sp_r[i] = sp_a[m][idx_S].real(); Sp_i[i] = sp_a[m][idx_S].imag();
                    Sm_r[i] = sm_a[m][idx_S].real(); Sm_i[i] = sm_a[m][idx_S].imag();
                }
            }
            if (!any_active) {
                continue;
            }

            for (int i=0; i < n_eff; ++i) {
                if (!active[i]) {
                    dp1[i] = 0.0;
                    dm1[i] = 0.0;
                    dp2[i] = 0.0;
                    dm2[i] = 0.0;
                    continue;
                }
                dp1[i] = wigner_d_element_scalar(l0, m, -spin, th_ptr[i]);
                if(spin_active) dm1[i] = wigner_d_element_scalar(l0, m, spin, th_ptr[i]);
                else dm1[i] = dp1[i];
                dp2[i] = 0.0;
                dm2[i] = 0.0;
            }

            auto accumulate_l_internal = [&](int64_t l, const double* __restrict__ d_p, const double* __restrict__ d_m) {
                double r_p = 0, i_p = 0, r_m = 0, i_m = 0;
                const double parity = ((l+m)%2 == 0) ? 1.0 : -1.0;
                #pragma clang loop vectorize(enable)
                for(int i=0; i<n_pairs; ++i) {
                    r_p += (Np_r[i]*weightsN[i]*d_p[i] + Sp_r[i]*weightsS[i]*d_m[i]*parity);
                    i_p += (Np_i[i]*weightsN[i]*d_p[i] + Sp_i[i]*weightsS[i]*d_m[i]*parity);
                    r_m += (Nm_r[i]*weightsN[i]*d_m[i] + Sm_r[i]*weightsS[i]*d_p[i]*parity);
                    i_m += (Nm_i[i]*weightsN[i]*d_m[i] + Sm_i[i]*weightsS[i]*d_p[i]*parity);
                }
                if (has_equator && active[n_pairs]) {
                    int i = n_pairs;
                    r_p += Np_r[i]*weightsN[i]*d_p[i]; i_p += Np_i[i]*weightsN[i]*d_p[i];
                    r_m += Nm_r[i]*weightsN[i]*d_m[i]; i_m += Nm_i[i]*weightsN[i]*d_m[i];
                }
                const double sc = pre_phase * precomputed_l_factors[l];
                cp_out[idx_base+(l-m)] = c10::complex<double>(r_p*sc, i_p*sc);
                cm_out[idx_base+(l-m)] = c10::complex<double>(r_m*sc, i_m*sc);
            };

            accumulate_l_internal(l0, dp1, dm1);

            if (l0 < lmax) {
                for (int i=0; i < n_eff; ++i) {
                    if (!active[i]) {
                        dpc[i] = 0.0;
                        dmc[i] = 0.0;
                        continue;
                    }
                    dpc[i] = wigner_d_element_scalar(l0+1, m, -spin, th_ptr[i]);
                    if(spin_active) dmc[i] = wigner_d_element_scalar(l0+1, m, spin, th_ptr[i]);
                    else dmc[i] = dpc[i];
                }
                accumulate_l_internal(l0+1, dpc, dmc);
                { auto t = dp2; dp2 = dp1; dp1 = dpc; dpc = t; }
                { auto t = dm2; dm2 = dm1; dm1 = dmc; dmc = t; }

                for (int64_t l = l0 + 2; l <= lmax; ++l) {
                    const auto& c = lut_ptr[l - (l0 + 2)];
                    const double c1 = c.c1, c2 = c.c2_p, c3 = c.c3;
                    const double parity = ((l+m)%2 == 0) ? 1.0 : -1.0;
                    double r_p = 0, i_p = 0, r_m = 0, i_m = 0;

                    #pragma clang loop vectorize(enable)
                    for (int i=0; i < n_pairs; ++i) {
                        const double vpN = (c1*ct_ptr[i] + c2)*dp1[i] + c3*dp2[i];
                        const double vmN = (c1*ct_ptr[i] - c2)*dm1[i] + c3*dm2[i];
                        dpc[i] = vpN; dmc[i] = vmN;
                        r_p += (Np_r[i]*weightsN[i]*vpN + Sp_r[i]*weightsS[i]*vmN*parity);
                        i_p += (Np_i[i]*weightsN[i]*vpN + Sp_i[i]*weightsS[i]*vmN*parity);
                        r_m += (Nm_r[i]*weightsN[i]*vmN + Sm_r[i]*weightsS[i]*vpN*parity);
                        i_m += (Nm_i[i]*weightsN[i]*vmN + Sm_i[i]*weightsS[i]*vpN*parity);
                    }
                    if (has_equator && active[n_pairs]) {
                        int i = n_pairs;
                        const double vpN = (c1*ct_ptr[i] + c2)*dp1[i] + c3*dp2[i];
                        const double vmN = (c1*ct_ptr[i] - c2)*dm1[i] + c3*dm2[i];
                        dpc[i] = vpN; dmc[i] = vmN;
                        r_p += Np_r[i]*weightsN[i]*vpN; i_p += Np_i[i]*weightsN[i]*vpN;
                        r_m += Nm_r[i]*weightsN[i]*vmN; i_m += Nm_i[i]*weightsN[i]*vmN;
                    }
                    const double sc = pre_phase * precomputed_l_factors[l];
                    cp_out[idx_base+(l-m)] = c10::complex<double>(r_p*sc, i_p*sc);
                    cm_out[idx_base+(l-m)] = c10::complex<double>(r_m*sc, i_m*sc);
                    { auto t = dp2; dp2 = dp1; dp1 = dpc; dpc = t; }
                    { auto t = dm2; dm2 = dm1; dm1 = dmc; dmc = t; }
                }
            }
        }
    });

    return torch::stack({alms_plus, alms_minus}, 0);
}

torch::Tensor healpix_spin_interpolate_recurrence_cpu(
    const torch::Tensor& alms,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin
) {
    auto alms_f = alms.contiguous();
    auto alms_a = alms_f.accessor<c10::complex<double>, 2>();
    int64_t nrings = theta.size(0);
    int64_t n_eff = (nrings + 1) / 2;
    int64_t n_pairs = nrings / 2;
    bool has_equator = (nrings % 2 != 0);
    ensure_l_factors(lmax);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor s_modes = torch::zeros({2, mmax + 1, nrings}, options);
    auto s_modes_a = s_modes.accessor<c10::complex<double>, 3>();
    auto th_ptr = theta.data_ptr<double>();
    
    std::vector<int64_t> m_offsets(mmax + 1);
    int64_t current_lut_off = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        m_offsets[m] = current_lut_off;
        int64_t l0 = std::max((int64_t)std::abs(spin), m);
        if (lmax >= l0 + 2) {
            current_lut_off += (lmax - (l0 + 2) + 1);
        }
    }
    const auto& lut = g_recurrence_cache.get_coeffs(lmax, mmax, spin, m_offsets);

    std::vector<double> cos_theta_all(n_eff);
    for(int i=0; i<n_eff; ++i) cos_theta_all[i] = std::cos(th_ptr[i]);
    std::vector<int64_t> mlim_north(n_eff, lmax);
    if (spin_mlim_prune_enabled()) {
        const int64_t spin_abs = std::abs(spin);
        for (int64_t i = 0; i < n_eff; ++i) {
            const double cth = cos_theta_all[static_cast<size_t>(i)];
            const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
            mlim_north[static_cast<size_t>(i)] = sharp_mlim_estimate(lmax, spin_abs, sth, cth);
        }
    }

    // For small N, overhead of parallel_for is high. Use a larger grain size (e.g. 8).
    at::parallel_for(0, mmax + 1, 8, [&](int64_t m_start, int64_t m_end) {
        // Allocate working buffers ONCE per thread task, rather than per inner loop.
        std::vector<double> v_dp2(n_eff), v_dp1(n_eff), v_dpc(n_eff);
        std::vector<double> v_dm2(n_eff), v_dm1(n_eff), v_dmc(n_eff);
        std::vector<double> v_rNp(n_eff), v_iNp(n_eff), v_rNm(n_eff), v_iNm(n_eff);
        std::vector<double> v_rSp(n_eff), v_iSp(n_eff), v_rSm(n_eff), v_iSm(n_eff);
        std::vector<uint8_t> v_active(n_eff);
        double * __restrict__ dp2 = v_dp2.data();
        double * __restrict__ dp1 = v_dp1.data();
        double * __restrict__ dpc = v_dpc.data();
        double * __restrict__ dm2 = v_dm2.data();
        double * __restrict__ dm1 = v_dm1.data();
        double * __restrict__ dmc = v_dmc.data();
        double * __restrict__ rNp = v_rNp.data(); double * __restrict__ iNp = v_iNp.data();
        double * __restrict__ rNm = v_rNm.data(); double * __restrict__ iNm = v_iNm.data();
        double * __restrict__ rSp = v_rSp.data(); double * __restrict__ iSp = v_iSp.data();
        double * __restrict__ rSm = v_rSm.data(); double * __restrict__ iSm = v_iSm.data();
        uint8_t * __restrict__ active = v_active.data();
        const double * __restrict__ ct_ptr = cos_theta_all.data();

        for (int64_t m = m_start; m < m_end; ++m) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            int64_t idx_base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const double pre_phase = ((m % 2 != 0) ? -1.0 : 1.0) * ((std::abs(spin) % 2 != 0) ? -1.0 : 1.0);
            const bool spin_active = (spin != 0);
            const auto* lut_ptr = &lut[m_offsets[m]];
            bool any_active = false;

            for(int i=0; i<n_eff; ++i) {
                const bool on = (m <= mlim_north[static_cast<size_t>(i)]);
                active[i] = static_cast<uint8_t>(on ? 1 : 0);
                any_active = any_active || on;
                rNp[i]=0; iNp[i]=0; rNm[i]=0; iNm[i]=0;
                rSp[i]=0; iSp[i]=0; rSm[i]=0; iSm[i]=0;
            }
            if (!any_active) {
                continue;
            }

            for(int i=0; i<n_eff; ++i) {
                if (!active[i]) {
                    dp1[i] = 0.0;
                    dm1[i] = 0.0;
                    dp2[i] = 0.0;
                    dm2[i] = 0.0;
                    continue;
                }
                dp1[i] = wigner_d_element_scalar(l0, m, -spin, th_ptr[i]);
                if(spin_active) dm1[i] = wigner_d_element_scalar(l0, m, spin, th_ptr[i]);
                else dm1[i] = dp1[i];
                dp2[i] = 0.0;
                dm2[i] = 0.0;
            }


            auto update_block_internal = [&](int64_t l, const double* __restrict__ d_p, const double* __restrict__ d_m) {
                const c10::complex<double> a0 = alms_a[0][idx_base+(l-m)];
                const c10::complex<double> a1 = spin_active ? alms_a[1][idx_base+(l-m)] : a0;
                const double sc = pre_phase * precomputed_l_factors[l];
                const double r0 = a0.real()*sc, i0 = a0.imag()*sc, r1 = a1.real()*sc, i1 = a1.imag()*sc;
                const double parity = ((l+m)%2 == 0) ? 1.0 : -1.0;

                #pragma clang loop vectorize(enable)
                for(int i=0; i<n_pairs; ++i) {
                    rNp[i] += r0*d_p[i]; iNp[i] += i0*d_p[i];
                    rNm[i] += r1*d_m[i]; iNm[i] += i1*d_m[i];
                    rSp[i] += r0*d_m[i]*parity; iSp[i] += i0*d_m[i]*parity;
                    rSm[i] += r1*d_p[i]*parity; iSm[i] += i1*d_p[i]*parity;
                }
                if (has_equator && active[n_pairs]) {
                    int i = n_pairs;
                    rNp[i] += r0*d_p[i]; iNp[i] += i0*d_p[i];
                    rNm[i] += r1*d_m[i]; iNm[i] += i1*d_m[i];
                }
            };

            update_block_internal(l0, dp1, dm1);
            if (l0 < lmax) {
                for(int i=0; i<n_eff; ++i) {
                    if (!active[i]) {
                        dpc[i] = 0.0;
                        dmc[i] = 0.0;
                        continue;
                    }
                    dpc[i] = wigner_d_element_scalar(l0+1, m, -spin, th_ptr[i]);
                    if(spin_active) dmc[i] = wigner_d_element_scalar(l0+1, m, spin, th_ptr[i]);
                    else dmc[i] = dpc[i];
                }
                update_block_internal(l0+1, dpc, dmc);
                { auto t = dp2; dp2 = dp1; dp1 = dpc; dpc = t; }
                { auto t = dm2; dm2 = dm1; dm1 = dmc; dmc = t; }

                for (int64_t l = l0 + 2; l <= lmax; ++l) {
                    const auto& c = lut_ptr[l - (l0 + 2)];
                    const c10::complex<double> a0 = alms_a[0][idx_base+(l-m)];
                    const c10::complex<double> a1 = spin_active ? alms_a[1][idx_base+(l-m)] : a0;
                    const double sc = pre_phase * precomputed_l_factors[l];
                    const double r0 = a0.real()*sc, i0 = a0.imag()*sc, r1 = a1.real()*sc, i1 = a1.imag()*sc;
                    const double parity = ((l+m)%2 == 0) ? 1.0 : -1.0;
                    const double c1 = c.c1, c2 = c.c2_p, c3 = c.c3;

                    #pragma clang loop vectorize(enable)
                    for(int i=0; i<n_pairs; ++i) {
                        const double vpN = (c1*ct_ptr[i] + c2)*dp1[i] + c3*dp2[i];
                        const double vmN = (c1*ct_ptr[i] - c2)*dm1[i] + c3*dm2[i];
                        dpc[i] = vpN; dmc[i] = vmN;
                        rNp[i] += r0*vpN; iNp[i] += i0*vpN;
                        rNm[i] += r1*vmN; iNm[i] += i1*vmN;
                        rSp[i] += r0*vmN*parity; iSp[i] += i0*vmN*parity;
                        rSm[i] += r1*vpN*parity; iSm[i] += i1*vpN*parity;
                    }
                    if (has_equator && active[n_pairs]) {
                        int i = n_pairs;
                        const double vpN = (c1*ct_ptr[i] + c2)*dp1[i] + c3*dp2[i];
                        const double vmN = (c1*ct_ptr[i] - c2)*dm1[i] + c3*dm2[i];
                        dpc[i] = vpN; dmc[i] = vmN;
                        rNp[i] += r0*vpN; iNp[i] += i0*vpN;
                        rNm[i] += r1*vmN; iNm[i] += i1*vmN;
                    }
                    { auto t = dp2; dp2 = dp1; dp1 = dpc; dpc = t; }
                    { auto t = dm2; dm2 = dm1; dm1 = dmc; dmc = t; }
                }
            }
            
            for(int i=0; i<n_eff; ++i) {
                if (!active[i]) {
                    continue;
                }
                int64_t idx_N = i;
                s_modes_a[0][m][idx_N] = c10::complex<double>(rNp[i], iNp[i]);
                s_modes_a[1][m][idx_N] = c10::complex<double>(rNm[i], iNm[i]);
                if (i < n_pairs) {
                    int64_t idx_S = nrings - 1 - i;
                    s_modes_a[0][m][idx_S] = c10::complex<double>(rSp[i], iSp[i]);
                    s_modes_a[1][m][idx_S] = c10::complex<double>(rSm[i], iSm[i]);
                }
            }
        }
    });

    return s_modes;
}

torch::Tensor healpix_query_disc_cpu(

    int64_t nside,
    const torch::Tensor& vec_xyz,
    double cos_lim,
    bool nest
) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!vec_xyz.device().is_cpu() || vec_xyz.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_query_disc_cpu expects float64 CPU vec");
    }
    auto vec = vec_xyz.contiguous().reshape({-1});
    if (vec.numel() != 3) {
        throw std::runtime_error("vec must have length 3");
    }

    double vx = vec.data_ptr<double>()[0];
    double vy = vec.data_ptr<double>()[1];
    double vz = vec.data_ptr<double>()[2];
    const double vnorm = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (!(vnorm > 0.0)) {
        throw std::runtime_error("vec norm must be positive");
    }
    vx /= vnorm;
    vy /= vnorm;
    vz /= vnorm;

    const int64_t npix = 12 * nside * nside;
    if (cos_lim <= -1.0) {
        return torch::arange(npix, torch::TensorOptions().dtype(torch::kInt64));
    }
    if (cos_lim > 1.0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
    }

    const double vxy = std::sqrt(std::max(0.0, vx * vx + vy * vy));
    const double phi0 = wrap_rad_2pi(std::atan2(vy, vx));
    const double eps = 1.0e-14;
    struct Range {
        int64_t start;
        int64_t len;
    };
    std::vector<Range> ranges;
    ranges.reserve(static_cast<size_t>(std::max<int64_t>(64, nside * 4)));

    auto append_range = [&](int64_t startpix, int64_t ringpix, int64_t k0, int64_t k1) {
        if (k1 < 0 || k0 >= ringpix || k0 > k1) {
            return;
        }
        if (k0 < 0) {
            k0 = 0;
        }
        if (k1 >= ringpix) {
            k1 = ringpix - 1;
        }
        if (k0 > k1) {
            return;
        }
        const int64_t start = startpix + k0;
        const int64_t len = k1 - k0 + 1;
        if (!ranges.empty()) {
            auto& prev = ranges.back();
            const int64_t prev_end = prev.start + prev.len;
            if (start <= prev_end) {
                const int64_t new_end = std::max(prev_end, start + len);
                prev.len = new_end - prev.start;
                return;
            }
        }
        ranges.push_back(Range{start, len});
    };

    for (int64_t ring = 1; ring < 4 * nside; ++ring) {
        int64_t startpix = 0;
        int64_t ringpix = 0;
        double theta = 0.0;
        bool shifted = false;
        get_ring_info2_scalar(nside, ring, startpix, ringpix, theta, shifted);

        double z = std::cos(theta);
        if (z > 1.0) {
            z = 1.0;
        } else if (z < -1.0) {
            z = -1.0;
        }
        const double rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double dot_center = z * vz;
        const double span = rxy * vxy;

        if (span <= eps) {
            if (dot_center >= cos_lim) {
                append_range(startpix, ringpix, 0, ringpix - 1);
            }
            continue;
        }
        if (dot_center + span < cos_lim) {
            continue;
        }
        if (dot_center - span >= cos_lim) {
            append_range(startpix, ringpix, 0, ringpix - 1);
            continue;
        }

        double c = (cos_lim - dot_center) / span;
        if (c <= -1.0) {
            append_range(startpix, ringpix, 0, ringpix - 1);
            continue;
        }
        if (c > 1.0) {
            continue;
        }
        if (c > 1.0 - 1.0e-15) {
            c = 1.0;
        } else if (c < -1.0 + 1.0e-15) {
            c = -1.0;
        }
        const double alpha = std::acos(c);
        if (alpha >= (PI - 1.0e-15)) {
            append_range(startpix, ringpix, 0, ringpix - 1);
            continue;
        }

        const double dphi = TWO_PI / static_cast<double>(ringpix);
        const double phase = shifted ? 0.5 : 0.0;
        const double lo = wrap_rad_2pi(phi0 - alpha);
        const double hi = wrap_rad_2pi(phi0 + alpha);

        auto interval_to_k = [&](double a, double b, int64_t& k0, int64_t& k1) {
            k0 = static_cast<int64_t>(std::ceil(a / dphi - phase - eps));
            k1 = static_cast<int64_t>(std::floor(b / dphi - phase + eps));
        };

        if (lo <= hi) {
            int64_t k0 = 0;
            int64_t k1 = -1;
            interval_to_k(lo, hi, k0, k1);
            append_range(startpix, ringpix, k0, k1);
        } else {
            // Wrap-around interval: emit low segment first to keep ascending ring order.
            int64_t k0a = 0;
            int64_t k1a = -1;
            interval_to_k(0.0, hi, k0a, k1a);
            append_range(startpix, ringpix, k0a, k1a);

            int64_t k0b = 0;
            int64_t k1b = -1;
            interval_to_k(lo, TWO_PI - eps, k0b, k1b);
            append_range(startpix, ringpix, k0b, k1b);
        }
    }

    if (ranges.empty()) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
    }

    int64_t total = 0;
    for (const auto& r : ranges) {
        total += r.len;
    }
    auto ring_out = torch::empty({total}, torch::TensorOptions().dtype(torch::kInt64));
    auto* ptr = ring_out.data_ptr<int64_t>();
    int64_t off = 0;
    for (const auto& r : ranges) {
        std::iota(ptr + off, ptr + off + r.len, r.start);
        off += r.len;
    }
    if (!nest) {
        return ring_out;
    }
    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t nl2 = 2 * nside;
    const int64_t nside_shift = ilog2_pow2(nside);
    const int64_t four_nside_shift = nside_shift + 2;
    const int64_t four_nside_mask = (4 * nside) - 1;

    std::vector<int64_t> nest_pix(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        const int64_t p = ptr[i];
        int64_t ix;
        int64_t iy;
        int64_t face_num;
        ring_to_xyf(
            nside,
            ncap,
            npix,
            nl2,
            nside_shift,
            four_nside_shift,
            four_nside_mask,
            p,
            ix,
            iy,
            face_num
        );
        nest_pix[static_cast<size_t>(i)] = xyf_to_nest(npface, ix, iy, face_num);
    }
    std::sort(nest_pix.begin(), nest_pix.end());
    return torch::from_blob(
        nest_pix.data(),
        {total},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone();
}

}  // namespace torchfits
