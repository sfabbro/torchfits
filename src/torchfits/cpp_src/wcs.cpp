#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace torchfits {

constexpr double WCS_PI = 3.14159265358979323846;
constexpr double D2R = WCS_PI / 180.0;
constexpr double R2D = 180.0 / WCS_PI;
constexpr double SQRT2 = 1.4142135623730950488;

namespace {

struct TpvAffineSeedParams {
    double b1 = 0.0;
    double b2 = 0.0;
    double a11 = 0.0;
    double a12 = 0.0;
    double a21 = 0.0;
    double a22 = 0.0;
    bool valid = false;
};

using TpvPowCache = std::array<torch::Tensor, 8>;

TpvPowCache tpv_make_pow_cache(const torch::Tensor& base) {
    TpvPowCache pows;
    pows[0] = torch::ones_like(base);
    pows[1] = base;
    for (int i = 2; i <= 7; ++i) {
        pows[i] = pows[i - 1] * base;
    }
    return pows;
}

TpvAffineSeedParams tpv_extract_affine_seed_params(
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    TpvAffineSeedParams p;
    auto idx1_acc = idx1.accessor<int64_t, 2>();
    auto c1_acc = c1.accessor<double, 1>();
    const int64_t n1 = c1.size(0);
    for (int64_t k = 0; k < n1; ++k) {
        const int64_t px = idx1_acc[k][0];
        const int64_t py = idx1_acc[k][1];
        const int64_t pr = idx1_acc[k][2];
        const double coeff = c1_acc[k];
        if (pr != 0) continue;
        if (px == 0 && py == 0) p.b1 += coeff;
        else if (px == 1 && py == 0) p.a11 += coeff;
        else if (px == 0 && py == 1) p.a12 += coeff;
    }

    // Axis-2 TPV convention is swapped in evaluation: idx2 (px, py) corresponds to
    // eta(v, u), so px is power of v and py is power of u.
    auto idx2_acc = idx2.accessor<int64_t, 2>();
    auto c2_acc = c2.accessor<double, 1>();
    const int64_t n2 = c2.size(0);
    for (int64_t k = 0; k < n2; ++k) {
        const int64_t px = idx2_acc[k][0];
        const int64_t py = idx2_acc[k][1];
        const int64_t pr = idx2_acc[k][2];
        const double coeff = c2_acc[k];
        if (pr != 0) continue;
        if (px == 0 && py == 0) p.b2 += coeff;
        else if (px == 0 && py == 1) p.a21 += coeff;  // u coefficient
        else if (px == 1 && py == 0) p.a22 += coeff;  // v coefficient
    }

    const double det = p.a11 * p.a22 - p.a12 * p.a21;
    p.valid = std::isfinite(det) && std::abs(det) > 1e-18;
    return p;
}

std::pair<torch::Tensor, torch::Tensor> tpv_initial_guess_affine(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    auto p = tpv_extract_affine_seed_params(idx1, c1, idx2, c2);
    if (!p.valid) {
        return {xi_target.clone(), eta_target.clone()};
    }

    const double det = p.a11 * p.a22 - p.a12 * p.a21;
    const double inv11 = p.a22 / det;
    const double inv12 = -p.a12 / det;
    const double inv21 = -p.a21 / det;
    const double inv22 = p.a11 / det;

    auto dx = xi_target - p.b1;
    auto dy = eta_target - p.b2;
    auto u = dx * inv11 + dy * inv12;
    auto v = dx * inv21 + dy * inv22;
    return {u, v};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tpv_eval_axis_with_jacobian(
    const torch::Tensor& base_u,
    const torch::Tensor& idx,
    const torch::Tensor& coeffs,
    const TpvPowCache& xc,
    const TpvPowCache& yc,
    const TpvPowCache& r_p,
    const torch::Tensor& dr_dx,
    const torch::Tensor& dr_dy
) {
    auto out = torch::zeros_like(base_u);
    auto d_out_dx = torch::zeros_like(base_u);
    auto d_out_dy = torch::zeros_like(base_u);
    if (coeffs.numel() == 0) {
        return std::make_tuple(out, d_out_dx, d_out_dy);
    }

    auto radial_pref = torch::zeros_like(base_u);
    bool has_radial = false;

    auto idx_acc = idx.accessor<int64_t, 2>();
    auto c_acc = coeffs.accessor<double, 1>();
    const int64_t n_terms = coeffs.size(0);
    for (int64_t k = 0; k < n_terms; ++k) {
        const int64_t px = idx_acc[k][0];
        const int64_t py = idx_acc[k][1];
        const int64_t pr = idx_acc[k][2];
        const double coeff = c_acc[k];

        auto xy = xc[px] * yc[py];
        out.add_(xy * r_p[pr], coeff);

        if (px > 0) {
            d_out_dx.add_(xc[px - 1] * yc[py] * r_p[pr], coeff * static_cast<double>(px));
        }
        if (py > 0) {
            d_out_dy.add_(xc[px] * yc[py - 1] * r_p[pr], coeff * static_cast<double>(py));
        }
        if (pr > 0) {
            radial_pref.add_(xy * r_p[pr - 1], coeff * static_cast<double>(pr));
            has_radial = true;
        }
    }

    if (has_radial) {
        d_out_dx.add_(radial_pref * dr_dx);
        d_out_dy.add_(radial_pref * dr_dy);
    }

    return std::make_tuple(out, d_out_dx, d_out_dy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tpv_distort_and_jacobian_impl(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    auto r = (u * u + v * v).sqrt();
    auto x_p = tpv_make_pow_cache(u);
    auto y_p = tpv_make_pow_cache(v);
    auto r_p = tpv_make_pow_cache(r);

    auto r_safe = torch::where(r > 0, r, torch::ones_like(r));
    auto dr_du = torch::where(r > 0, u / r_safe, torch::zeros_like(u));
    auto dr_dv = torch::where(r > 0, v / r_safe, torch::zeros_like(v));

    auto [xi, dxi_du, dxi_dv] = tpv_eval_axis_with_jacobian(
        u, idx1, c1, x_p, y_p, r_p, dr_du, dr_dv
    );
    auto [eta, deta_dv, deta_du] = tpv_eval_axis_with_jacobian(
        u, idx2, c2, y_p, x_p, r_p, dr_dv, dr_du
    );

    return std::make_tuple(xi, eta, dxi_du, dxi_dv, deta_du, deta_dv);
}

std::pair<torch::Tensor, torch::Tensor> wcs_inverse_spherical_rotation_pole(
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg,
    double phi_p_rad,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z
) {
    auto ra_rad = ra_deg * D2R;
    auto dec_rad = dec_deg * D2R;
    auto sin_dec = dec_rad.sin();
    auto cos_dec = dec_rad.cos();
    auto sin_ra = ra_rad.sin();
    auto cos_ra = ra_rad.cos();

    auto X = cos_dec * cos_ra;
    auto Y = cos_dec * sin_ra;
    auto Z = sin_dec;

    auto u = X * east_x + Y * east_y;
    auto n = X * north_x + Y * north_y + Z * north_z;
    auto w = X * radial_x + Y * radial_y + Z * radial_z;
    w = w.clamp(-1.0, 1.0);

    auto theta = w.asin() * R2D;
    auto phi = (torch::atan2(u, n) + phi_p_rad) * R2D;
    phi = torch::remainder(phi, 360.0);
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_tan_intermediate_from_radec(
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z
) {
    auto ra_rad = ra_deg * D2R;
    auto dec_rad = dec_deg * D2R;
    auto sin_dec = dec_rad.sin();
    auto cos_dec = dec_rad.cos();
    auto sin_ra = ra_rad.sin();
    auto cos_ra = ra_rad.cos();

    auto X = cos_dec * cos_ra;
    auto Y = cos_dec * sin_ra;
    auto Z = sin_dec;

    auto u = X * east_x + Y * east_y;
    auto v = X * north_x + Y * north_y + Z * north_z;
    auto w = X * radial_x + Y * radial_y + Z * radial_z;
    auto scale = R2D / torch::clamp(w, 1e-12);
    return {u * scale, v * scale};
}
}

std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code
) {
    auto r = (xi_deg * xi_deg + eta_deg * eta_deg).sqrt();
    auto r_rad = r * D2R;
    auto phi = torch::atan2(xi_deg, -eta_deg) * R2D;
    
    torch::Tensor theta;
    
    if (proj_code == "TAN") {
        theta = torch::atan2(torch::ones_like(r_rad), r_rad) * R2D;
    } else if (proj_code == "SIN") {
        auto r_dim = r_rad.clamp(-1.0, 1.0);
        theta = r_dim.acos() * R2D;
    } else if (proj_code == "ARC") {
        theta = 90.0 - r;
    } else if (proj_code == "ZEA") {
        auto val = (r_rad / 2.0).clamp(-1.0, 1.0);
        theta = 90.0 - 2.0 * val.asin() * R2D;
    } else if (proj_code == "STG") {
        theta = 90.0 - 2.0 * (r_rad / 2.0).atan() * R2D;
    } else {
        throw std::runtime_error("Unknown zenithal projection: " + proj_code);
    }
    
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    const std::string& proj_code
) {
    torch::Tensor r;
    auto theta_rad = theta_deg * D2R;
    
    if (proj_code == "TAN") {
        auto tan_theta = theta_rad.tan();
        r = R2D / (tan_theta + 1e-12);
    } else if (proj_code == "SIN") {
        r = theta_rad.cos() * R2D;
    } else if (proj_code == "ARC") {
        r = 90.0 - theta_deg;
    } else if (proj_code == "ZEA") {
        r = 2.0 * ((90.0 - theta_deg) * D2R / 2.0).sin() * R2D;
    } else if (proj_code == "STG") {
        r = 2.0 * ((90.0 - theta_deg) * D2R / 2.0).tan() * R2D;
    } else {
        throw std::runtime_error("Unknown zenithal projection: " + proj_code);
    }
    
    auto phi_rad = phi_deg * D2R;
    auto xi = r * phi_rad.sin();
    auto eta = -r * phi_rad.cos();
    
    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_cylindrical_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code,
    double lambda_param
) {
    torch::Tensor phi, theta;
    
    if (proj_code == "CEA") {
        phi = xi_deg;
        auto val = (lambda_param * eta_deg * D2R).clamp(-1.0, 1.0);
        theta = val.asin() * R2D;
    } else if (proj_code == "MER") {
        phi = xi_deg;
        auto arg = (eta_deg * D2R).clamp(-20.0, 20.0);
        auto exp_val = arg.exp();
        theta = 2.0 * (exp_val.atan() * R2D - 45.0);
    } else {
        throw std::runtime_error("Unknown cylindrical projection: " + proj_code);
    }
    
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_cylindrical_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    const std::string& proj_code,
    double lambda_param
) {
    torch::Tensor xi, eta;
    
    if (proj_code == "CEA") {
        xi = phi_deg;
        eta = (R2D / lambda_param) * (theta_deg * D2R).sin();
    } else if (proj_code == "MER") {
        xi = phi_deg;
        auto half_theta = theta_deg / 2.0;
        auto tan_arg = ((45.0 + half_theta) * D2R).tan().clamp_min(1e-12);
        eta = R2D * tan_arg.log();
    } else {
        throw std::runtime_error("Unknown cylindrical projection: " + proj_code);
    }
    
    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_ait_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
) {
    auto X = xi_deg * D2R;
    auto Y = eta_deg * D2R;

    auto xq = X * 0.25;
    auto yq = Y * 0.5;
    auto r2 = xq * xq + yq * yq;
    auto valid = r2 <= 1.0;

    auto z = (1.0 - r2.clamp_max(1.0)).sqrt();
    auto z2 = z * z;
    auto phi_rad = 2.0 * torch::atan2(0.5 * z * X, 2.0 * z2 - 1.0);
    auto sin_theta = (z * Y).clamp(-1.0, 1.0);
    auto theta_rad = sin_theta.asin();

    auto phi = phi_rad * R2D;
    auto theta = theta_rad * R2D;

    auto invalid = ~valid;
    phi = phi.masked_fill(invalid, std::nan(""));
    theta = theta.masked_fill(invalid, std::nan(""));
    
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_ait_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg
) {
    auto phi_rad = phi_deg * D2R;
    auto theta_rad = theta_deg * D2R;
    
    auto half_phi = phi_rad / 2.0;
    auto cos_theta = theta_rad.cos();
    auto sin_theta = theta_rad.sin();
    auto cos_half_phi = half_phi.cos();
    
    auto denom = (0.5 * (1.0 + cos_theta * cos_half_phi)).sqrt();
    
    auto xi = 2.0 * cos_theta * half_phi.sin() / denom * R2D;
    auto eta = sin_theta / denom * R2D;
    
    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_mol_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
) {
    auto X = xi_deg * D2R;
    auto Y = eta_deg * D2R;
    
    auto valid = Y.abs() <= SQRT2;
    
    auto sin_gamma = (Y / SQRT2).clamp(-1.0, 1.0);
    auto gamma = sin_gamma.asin();
    auto cos_gamma = (1.0 - sin_gamma * sin_gamma).clamp_min(0.0).sqrt();

    auto t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / WCS_PI;
    t_val = t_val.clamp(-1.0, 1.0);
    auto theta_rad = t_val.asin();

    auto denom = 2.0 * SQRT2 * cos_gamma;
    auto good = denom.abs() >= 1e-12;
    auto denom_safe = torch::where(good, denom, torch::ones_like(denom));
    auto phi_rad = WCS_PI * X / denom_safe;
    phi_rad = torch::where(good, phi_rad, torch::zeros_like(phi_rad));
    
    auto phi = phi_rad * R2D;
    auto theta = theta_rad * R2D;
    
    auto invalid = ~valid;
    phi = phi.masked_fill(invalid, std::nan(""));
    theta = theta.masked_fill(invalid, std::nan(""));
    
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_mol_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg
) {
    auto phi_rad = phi_deg * D2R;
    auto theta_rad = theta_deg * D2R;
    
    auto sin_theta = theta_rad.sin();
    
    auto gamma = torch::zeros_like(theta_rad);
    for (int i = 0; i < 10; ++i) {
        auto f = 2.0 * gamma + (2.0 * gamma).sin() - WCS_PI * sin_theta;
        auto fp = 2.0 + 2.0 * (2.0 * gamma).cos();
        gamma = gamma - f / (fp + 1e-12);
    }
    
    auto cos_gamma = gamma.cos();
    auto sin_gamma = gamma.sin();
    
    auto xi = 2.0 * SQRT2 * phi_rad * cos_gamma / WCS_PI * R2D;
    auto eta = SQRT2 * sin_gamma * R2D;
    
    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_hpx_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    double H,
    double K
) {
    double eta_scale = 90.0 * (K / H);
    double eta_boundary = eta_scale * (2.0 / 3.0);
    double eta_pole = 90.0;
    double polar_denom = eta_pole - eta_boundary;

    auto abs_eta = eta_deg.abs();
    auto invalid_eta = abs_eta > eta_pole;
    auto mask_eq = abs_eta <= eta_boundary;
    auto mask_pol = (~mask_eq) & (~invalid_eta);

    auto s_theta_eq = (eta_deg / eta_scale).clamp(-1.0, 1.0);
    auto theta_eq = s_theta_eq.asin() * R2D;

    auto sigma = ((eta_pole - abs_eta) / polar_denom).clamp_min(0.0);
    auto s_theta_pol = (eta_deg.sign() * (1.0 - (sigma * sigma) / 3.0)).clamp(-1.0, 1.0);
    auto theta_pol = s_theta_pol.asin() * R2D;

    auto xc = ((xi_deg - 45.0) / 90.0).round() * 90.0 + 45.0;
    auto dx = xi_deg - xc;
    auto sigma_safe = torch::where(sigma.abs() < 1e-12, torch::ones_like(sigma), sigma);
    auto phi_pol = xc + dx / sigma_safe;

    auto invalid_x = dx.abs() > (45.0 * sigma + 1e-8);
    auto invalid = invalid_eta | (mask_pol & invalid_x);

    auto phi = torch::where(mask_eq, xi_deg, phi_pol);
    auto theta = torch::where(mask_eq, theta_eq, theta_pol);
    phi = phi.masked_fill(invalid, std::nan(""));
    theta = theta.masked_fill(invalid, std::nan(""));
    
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_hpx_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    double H,
    double K
) {
    auto s_theta = (theta_deg * D2R).sin();
    auto abs_s = s_theta.abs();

    double eta_scale = 90.0 * (K / H);
    double eta_boundary = eta_scale * (2.0 / 3.0);

    auto mask_eq = abs_s <= (2.0 / 3.0);
    auto sigma = (3.0 * (1.0 - abs_s)).clamp_min(0.0).sqrt();
    auto eta_eq = eta_scale * s_theta;
    auto eta_pol = s_theta.sign() * (90.0 - (90.0 - eta_boundary) * sigma);

    auto xc = ((phi_deg - 45.0) / 90.0).round() * 90.0 + 45.0;
    auto xi_pol = xc + sigma * (phi_deg - xc);

    auto xi = torch::where(mask_eq, phi_deg, xi_pol);
    auto eta = torch::where(mask_eq, eta_eq, eta_pol);

    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_tpv_invert(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2,
    int64_t max_iter,
    double tol
) {
    auto [u, v] = tpv_initial_guess_affine(xi_target, eta_target, idx1, c1, idx2, c2);
    torch::Tensor active_idx;
    bool has_active = false;
    const double tol2 = tol * tol;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        torch::Tensor u_a, v_a, xi_t_a, eta_t_a;
        if (!has_active) {
            u_a = u;
            v_a = v;
            xi_t_a = xi_target;
            eta_t_a = eta_target;
        } else {
            if (active_idx.numel() == 0) {
                break;
            }
            u_a = u.index_select(0, active_idx);
            v_a = v.index_select(0, active_idx);
            xi_t_a = xi_target.index_select(0, active_idx);
            eta_t_a = eta_target.index_select(0, active_idx);
        }

        auto [xi, eta, j11, j12, j21, j22] = tpv_distort_and_jacobian_impl(
            u_a, v_a, idx1, c1, idx2, c2
        );
        auto rx = xi - xi_t_a;
        auto ry = eta - eta_t_a;

        auto dist2 = rx * rx + ry * ry;
        auto keep_active = dist2 >= tol2;
        if (!keep_active.any().item<bool>()) {
            break;
        }

        auto det = j11 * j22 - j12 * j21;
        auto det_sign = torch::where(det >= 0, torch::ones_like(det), -torch::ones_like(det));
        det = torch::where(det.abs() < 1e-18, det_sign * 1e-18, det);

        auto du = -(j22 * rx - j12 * ry) / det;
        auto dv = -(-j21 * rx + j11 * ry) / det;
        du = du.clamp(-1.0, 1.0);
        dv = dv.clamp(-1.0, 1.0);
        auto u_next = u_a + du;
        auto v_next = v_a + dv;

        const bool keep_all = keep_active.all().item<bool>();
        if (!has_active) {
            if (keep_all) {
                u = u_next;
                v = v_next;
            } else {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                u = u_next;
                v = v_next;
                active_idx = next_local_idx;
                has_active = true;
            }
        } else {
            u.index_copy_(0, active_idx, u_next);
            v.index_copy_(0, active_idx, v_next);
            if (!keep_all) {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                active_idx = active_idx.index_select(0, next_local_idx);
            }
        }
    }

    return {u, v};
}

std::tuple<torch::Tensor, torch::Tensor, int64_t, std::vector<int64_t>, int64_t> wcs_tpv_invert_trace(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2,
    int64_t max_iter,
    double tol
) {
    auto [u, v] = tpv_initial_guess_affine(xi_target, eta_target, idx1, c1, idx2, c2);
    torch::Tensor active_idx;
    bool has_active = false;
    const double tol2 = tol * tol;
    std::vector<int64_t> active_counts;
    active_counts.reserve(static_cast<size_t>(std::max<int64_t>(max_iter, 0)));
    int64_t iterations_used = 0;
    int64_t final_active = 0;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        torch::Tensor u_a, v_a, xi_t_a, eta_t_a;
        if (!has_active) {
            u_a = u;
            v_a = v;
            xi_t_a = xi_target;
            eta_t_a = eta_target;
        } else {
            if (active_idx.numel() == 0) {
                final_active = 0;
                break;
            }
            u_a = u.index_select(0, active_idx);
            v_a = v.index_select(0, active_idx);
            xi_t_a = xi_target.index_select(0, active_idx);
            eta_t_a = eta_target.index_select(0, active_idx);
        }

        active_counts.push_back(static_cast<int64_t>(u_a.numel()));
        iterations_used += 1;

        auto [xi, eta, j11, j12, j21, j22] = tpv_distort_and_jacobian_impl(
            u_a, v_a, idx1, c1, idx2, c2
        );
        auto rx = xi - xi_t_a;
        auto ry = eta - eta_t_a;

        auto dist2 = rx * rx + ry * ry;
        auto keep_active = dist2 >= tol2;
        if (!keep_active.any().item<bool>()) {
            final_active = 0;
            break;
        }

        auto det = j11 * j22 - j12 * j21;
        auto det_sign = torch::where(det >= 0, torch::ones_like(det), -torch::ones_like(det));
        det = torch::where(det.abs() < 1e-18, det_sign * 1e-18, det);

        auto du = -(j22 * rx - j12 * ry) / det;
        auto dv = -(-j21 * rx + j11 * ry) / det;
        du = du.clamp(-1.0, 1.0);
        dv = dv.clamp(-1.0, 1.0);
        auto u_next = u_a + du;
        auto v_next = v_a + dv;

        const bool keep_all = keep_active.all().item<bool>();
        if (!has_active) {
            if (keep_all) {
                u = u_next;
                v = v_next;
                final_active = static_cast<int64_t>(u.numel());
            } else {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                u = u_next;
                v = v_next;
                active_idx = next_local_idx;
                has_active = true;
                final_active = static_cast<int64_t>(active_idx.numel());
            }
        } else {
            u.index_copy_(0, active_idx, u_next);
            v.index_copy_(0, active_idx, v_next);
            if (!keep_all) {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                active_idx = active_idx.index_select(0, next_local_idx);
            }
            final_active = static_cast<int64_t>(active_idx.numel());
        }
    }

    if (iterations_used == max_iter) {
        if (!has_active) {
            final_active = static_cast<int64_t>(u.numel());
        } else if (active_idx.defined()) {
            final_active = static_cast<int64_t>(active_idx.numel());
        }
    }

    return std::make_tuple(u, v, iterations_used, active_counts, final_active);
}

inline std::pair<double, double> eval_tpv_scalar(
    double u, double v,
    const torch::Tensor& idx1, const torch::Tensor& c1,
    const torch::Tensor& idx2, const torch::Tensor& c2) 
{
    if (c1.numel() == 0 && c2.numel() == 0) return {u, v};
    double r = std::sqrt(u * u + v * v);
    double xp[8] = {1.0, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u, u*u*u*u*u*u, u*u*u*u*u*u*u};
    double yp[8] = {1.0, v, v*v, v*v*v, v*v*v*v, v*v*v*v*v, v*v*v*v*v*v, v*v*v*v*v*v*v};
    double rp[8] = {1.0, r, r*r, r*r*r, r*r*r*r, r*r*r*r*r, r*r*r*r*r*r, r*r*r*r*r*r*r};
    
    double xi = 0.0;
    const int64_t* idx1_p = idx1.data_ptr<int64_t>();
    const double* c1_p = c1.data_ptr<double>();
    for (int k = 0; k < c1.size(0); ++k) {
        xi += xp[idx1_p[k*3]] * yp[idx1_p[k*3+1]] * rp[idx1_p[k*3+2]] * c1_p[k];
    }
    
    double eta = 0.0;
    const int64_t* idx2_p = idx2.data_ptr<int64_t>();
    const double* c2_p = c2.data_ptr<double>();
    for (int k = 0; k < c2.size(0); ++k) {
        eta += yp[idx2_p[k*3]] * xp[idx2_p[k*3+1]] * rp[idx2_p[k*3+2]] * c2_p[k];
    }
    return {xi, eta};
}

inline std::tuple<double, double, double, double, double, double> eval_tpv_jac_scalar(
    double u, double v,
    const torch::Tensor& idx1, const torch::Tensor& c1,
    const torch::Tensor& idx2, const torch::Tensor& c2)
{
    if (c1.numel() == 0 && c2.numel() == 0) return {u, v, 1.0, 0.0, 0.0, 1.0};
    double r = std::sqrt(u * u + v * v);
    double xp[8] = {1.0, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u, u*u*u*u*u*u, u*u*u*u*u*u*u};
    double yp[8] = {1.0, v, v*v, v*v*v, v*v*v*v, v*v*v*v*v, v*v*v*v*v*v, v*v*v*v*v*v*v};
    double rp[8] = {1.0, r, r*r, r*r*r, r*r*r*r, r*r*r*r*r, r*r*r*r*r*r, r*r*r*r*r*r*r};
    
    double dr_du = (r > 0) ? u / r : 0.0;
    double dr_dv = (r > 0) ? v / r : 0.0;

    auto eval_axis = [&](const torch::Tensor& idx, const torch::Tensor& c, const double* a_p, const double* b_p, double dr_da, double dr_db) {
        double out = 0, da = 0, db = 0, dr_pref = 0;
        const int64_t* idx_p = idx.data_ptr<int64_t>();
        const double* c_p = c.data_ptr<double>();
        for (int k = 0; k < c.size(0); ++k) {
            int64_t px = idx_p[k*3], py = idx_p[k*3+1], pr = idx_p[k*3+2];
            double coeff = c_p[k];
            out += a_p[px] * b_p[py] * rp[pr] * coeff;
            if (px > 0) da += a_p[px-1] * b_p[py] * rp[pr] * coeff * px;
            if (py > 0) db += a_p[px] * b_p[py-1] * rp[pr] * coeff * py;
            if (pr > 0) dr_pref += a_p[px] * b_p[py] * rp[pr-1] * coeff * pr;
        }
        da += dr_pref * dr_da;
        db += dr_pref * dr_db;
        return std::make_tuple(out, da, db);
    };

    auto [xi, j11, j12] = eval_axis(idx1, c1, xp, yp, dr_du, dr_dv);
    auto [eta, j22, j21] = eval_axis(idx2, c2, yp, xp, dr_dv, dr_du); // note: arguments swapped
    return {xi, eta, j11, j12, j21, j22};
}

inline std::pair<double, double> invert_tpv_scalar(
    double xi_target, double eta_target,
    const torch::Tensor& idx1, const torch::Tensor& c1,
    const torch::Tensor& idx2, const torch::Tensor& c2)
{
    if (c1.numel() == 0 && c2.numel() == 0) return {xi_target, eta_target};
    
    auto p = tpv_extract_affine_seed_params(idx1, c1, idx2, c2);
    double u, v;
    if (p.valid) {
        double det = p.a11 * p.a22 - p.a12 * p.a21;
        double inv11 = p.a22 / det, inv12 = -p.a12 / det;
        double inv21 = -p.a21 / det, inv22 = p.a11 / det;
        double dx = xi_target - p.b1, dy = eta_target - p.b2;
        u = dx * inv11 + dy * inv12;
        v = dx * inv21 + dy * inv22;
    } else {
        u = xi_target; v = eta_target;
    }
    
    double tol2 = 1e-22;
    for (int i = 0; i < 20; ++i) {
        auto [xi, eta, j11, j12, j21, j22] = eval_tpv_jac_scalar(u, v, idx1, c1, idx2, c2);
        double rx = xi - xi_target, ry = eta - eta_target;
        if (rx*rx + ry*ry < tol2) break;
        double det = j11 * j22 - j12 * j21;
        if (std::abs(det) < 1e-18) det = (det >= 0) ? 1e-18 : -1e-18;
        double du = -(j22 * rx - j12 * ry) / det;
        double dv = -(-j21 * rx + j11 * ry) / det;
        u += std::max(-1.0, std::min(1.0, du));
        v += std::max(-1.0, std::min(1.0, dv));
    }
    return {u, v};
}

void wcs_pixel_to_world_fused_cpu(
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor,
    torch::Tensor& ra_tensor,
    torch::Tensor& dec_tensor,
    double crpix0, double crpix1,
    double cd00, double cd01,
    double cd10, double cd11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x, double east_y,
    double north_x, double north_y, double north_z,
    double radial_x, double radial_y, double radial_z,
    double ra_p_rad, double dec_p_rad, double phi_p_rad,
    const std::string& native_type,
    double alpha0, double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
) {
    const int64_t n = x_tensor.numel();
    const double* x_ptr = x_tensor.data_ptr<double>();
    const double* y_ptr = y_tensor.data_ptr<double>();
    double* ra_ptr = ra_tensor.data_ptr<double>();
    double* dec_ptr = dec_tensor.data_ptr<double>();

    bool use_center_equator_fast = (native_type == "center" && std::abs(delta0) < 1e-12 && std::abs(std::fmod(phi_p_rad * R2D, 360.0)) < 1e-12);

    for (int64_t i = 0; i < n; ++i) {
        // 1. CD Matrix application
        double dx = x_ptr[i] - crpix0;
        double dy = y_ptr[i] - crpix1;
        double u_deg = cd00 * dx + cd01 * dy;
        double v_deg = cd10 * dx + cd11 * dy;

        double xi_deg = u_deg;
        double eta_deg = v_deg;

        if (is_tpv) {
            auto [tpv_xi, tpv_eta] = eval_tpv_scalar(u_deg, v_deg, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2);
            xi_deg = tpv_xi;
            eta_deg = tpv_eta;
        }

        // 2. Projection (xi, eta) -> (phi, theta)
        double phi, theta;
        if (proj_code == "TAN") {
            double r = std::sqrt(xi_deg * xi_deg + eta_deg * eta_deg);
            double r_rad = r * D2R;
            phi = std::atan2(-xi_deg, -eta_deg) * R2D;
            theta = std::atan2(1.0, std::max(r_rad, 1e-15)) * R2D;
        } else if (proj_code == "SIN") {
            double r = std::sqrt(xi_deg * xi_deg + eta_deg * eta_deg);
            double r_rad = std::min(r * D2R, 1.0);
            phi = std::atan2(-xi_deg, -eta_deg) * R2D;
            theta = std::acos(r_rad) * R2D;
        } else if (proj_code == "CEA") {
            phi = xi_deg;
            double val = std::max(-1.0, std::min(1.0, lambda_param * eta_deg * D2R));
            theta = std::asin(val) * R2D;
        } else if (proj_code == "AIT") {
            double X = xi_deg * D2R;
            double Y = eta_deg * D2R;
            double xq = X * 0.25;
            double yq = Y * 0.5;
            double r2 = xq * xq + yq * yq;
            if (r2 > 1.0) {
                phi = std::nan("");
                theta = std::nan("");
            } else {
                double z = std::sqrt(1.0 - r2);
                phi = 2.0 * std::atan2(0.5 * z * X, 2.0 * z * z - 1.0) * R2D;
                theta = std::asin(std::max(-1.0, std::min(1.0, z * Y))) * R2D;
            }
        } else if (proj_code == "HPX") {
            double eta_scale = 90.0 * (hpx_K / hpx_H);
            double eta_boundary = eta_scale * (2.0 / 3.0);
            double abs_eta = std::abs(eta_deg);
            if (abs_eta <= eta_boundary) {
                phi = xi_deg;
                theta = std::asin(std::max(-1.0, std::min(1.0, eta_deg / eta_scale))) * R2D;
            } else {
                double sigma = std::max(0.0, (90.0 - abs_eta) / (90.0 - eta_boundary));
                double xc = std::round((xi_deg - 45.0) / 90.0) * 90.0 + 45.0;
                double dx_hpx = xi_deg - xc;
                phi = xc + (sigma > 1e-12 ? dx_hpx / sigma : 0.0);
                double s_theta = (eta_deg >= 0 ? 1.0 : -1.0) * (1.0 - (sigma * sigma) / 3.0);
                theta = std::asin(std::max(-1.0, std::min(1.0, s_theta))) * R2D;
            }
        } else {
            phi = std::nan("");
            theta = std::nan("");
        }

        // 3. Spherical Rotation (phi, theta) -> (ra, dec)
        if (use_center_equator_fast) {
            ra_ptr[i] = std::fmod(phi + alpha0 + 360.0, 360.0);
            dec_ptr[i] = theta;
        } else {
            double phi_rad = phi * D2R;
            double theta_rad = theta * D2R;
            double cos_theta = std::cos(theta_rad);
            double sin_theta = std::sin(theta_rad);
            double cos_phi_diff = std::cos(phi_rad - phi_p_rad);

            double costhe_cosphi = cos_theta * cos_phi_diff;
            double arg = std::max(-1.0, std::min(1.0, sin_theta * std::sin(dec_p_rad) + costhe_cosphi * std::cos(dec_p_rad)));
            dec_ptr[i] = std::asin(arg) * R2D;

            double y_rot = cos_theta * std::sin(phi_rad - phi_p_rad);
            double x_rot = sin_theta * std::cos(dec_p_rad) - costhe_cosphi * std::sin(dec_p_rad);
            ra_ptr[i] = std::fmod(std::atan2(y_rot, x_rot) * R2D + ra_p_rad * R2D + 360.0, 360.0);
        }
    }
}

void wcs_world_to_pixel_fused_cpu(
    const torch::Tensor& ra_tensor,
    const torch::Tensor& dec_tensor,
    torch::Tensor& x_tensor,
    torch::Tensor& y_tensor,
    double crpix0, double crpix1,
    double cdi00, double cdi01,
    double cdi10, double cdi11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x, double east_y,
    double north_x, double north_y, double north_z,
    double radial_x, double radial_y, double radial_z,
    double ra_p_rad, double dec_p_rad, double phi_p_rad,
    const std::string& native_type,
    double alpha0, double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
) {
    const int64_t n = ra_tensor.numel();
    const double* ra_ptr = ra_tensor.data_ptr<double>();
    const double* dec_ptr = dec_tensor.data_ptr<double>();
    double* x_ptr = x_tensor.data_ptr<double>();
    double* y_ptr = y_tensor.data_ptr<double>();

    bool use_center_equator_fast = (native_type == "center" && std::abs(delta0) < 1e-12 && std::abs(std::fmod(phi_p_rad * R2D, 360.0)) < 1e-12);

    for (int64_t i = 0; i < n; ++i) {
        // 1. Inverse Spherical Rotation (ra, dec) -> (phi, theta)
        double ra_deg = ra_ptr[i];
        double dec_deg = dec_ptr[i];
        
        double phi, theta;
        if (use_center_equator_fast) {
            double phi_tmp = std::fmod(ra_deg - alpha0 + 180.0, 360.0);
            if (phi_tmp < 0) phi_tmp += 360.0;
            phi = phi_tmp - 180.0;
            theta = dec_deg;
        } else {
            double ra_rad = ra_deg * D2R;
            double dec_rad = dec_deg * D2R;
            double sin_dec = std::sin(dec_rad);
            double cos_dec = std::cos(dec_rad);
            double sin_ra = std::sin(ra_rad);
            double cos_ra = std::cos(ra_rad);

            double X = cos_dec * cos_ra;
            double Y = cos_dec * sin_ra;
            double Z = sin_dec;

            if (native_type == "pole") {
                double u_rot = X * east_x + Y * east_y;
                double n_rot = X * north_x + Y * north_y + Z * north_z;
                double w_rot = std::max(-1.0, std::min(1.0, X * radial_x + Y * radial_y + Z * radial_z));
                theta = std::asin(w_rot) * R2D;
                phi = std::atan2(u_rot, n_rot) * R2D + phi_p_rad * R2D;
                phi = std::fmod(phi + 360.0, 360.0);
            } else {
                double u_rot = X * east_x + Y * east_y;
                double v_rot = -(X * radial_x + Y * radial_y + Z * radial_z);
                double w_rot = std::max(-1.0, std::min(1.0, X * north_x + Y * north_y + Z * north_z));
                theta = std::asin(w_rot) * R2D;
                phi = std::atan2(u_rot, -v_rot) * R2D;
                phi = std::fmod(phi + 360.0, 360.0);
            }
        }

        // 2. Deprojection (phi, theta) -> (xi, eta)
        double xi, eta;
        double phi_w = std::fmod(phi + 180.0, 360.0);
        if (phi_w < 0) phi_w += 360.0;
        phi_w -= 180.0;

        if (proj_code == "TAN") {
            double theta_rad = theta * D2R;
            double r = R2D / std::tan(theta_rad);
            double phi_rad = phi * D2R;
            xi = -r * std::sin(phi_rad);
            eta = -r * std::cos(phi_rad);
        } else if (proj_code == "SIN") {
            double theta_rad = theta * D2R;
            double r = std::cos(theta_rad) * R2D;
            double phi_rad = phi * D2R;
            xi = -r * std::sin(phi_rad);
            eta = -r * std::cos(phi_rad);
        } else if (proj_code == "CEA") {
            xi = phi_w;
            eta = (R2D / lambda_param) * std::sin(theta * D2R);
        } else if (proj_code == "AIT") {
            double phi_rad = phi_w * D2R;
            double theta_rad = theta * D2R;
            double half_phi = phi_rad * 0.5;
            double cos_theta = std::cos(theta_rad);
            double sin_theta = std::sin(theta_rad);
            double cos_half_phi = std::cos(half_phi);
            double denom = std::sqrt(0.5 * (1.0 + cos_theta * cos_half_phi));
            xi = 2.0 * cos_theta * std::sin(half_phi) / denom * R2D;
            eta = sin_theta / denom * R2D;
        } else if (proj_code == "HPX") {
            double s_theta = std::sin(theta * D2R);
            double abs_s = std::abs(s_theta);
            double eta_scale = 90.0 * (hpx_K / hpx_H);
            double eta_boundary = eta_scale * (2.0 / 3.0);
            if (abs_s <= (2.0 / 3.0)) {
                xi = phi_w;
                eta = eta_scale * s_theta;
            } else {
                double sigma = std::sqrt(std::max(0.0, 3.0 * (1.0 - abs_s)));
                eta = (s_theta >= 0 ? 1.0 : -1.0) * (90.0 - (90.0 - eta_boundary) * sigma);
                double xc = std::round((phi_w - 45.0) / 90.0) * 90.0 + 45.0;
                xi = xc + sigma * (phi_w - xc);
            }
        } else {
            xi = std::nan("");
            eta = std::nan("");
        }

        double u_deg = xi;
        double v_deg = eta;

        if (is_tpv) {
            auto [tpv_u, tpv_v] = invert_tpv_scalar(xi, eta, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2);
            u_deg = tpv_u;
            v_deg = tpv_v;
        }

        // 3. Inverse CD Matrix
        x_ptr[i] = cdi00 * u_deg + cdi01 * v_deg + crpix0;
        y_ptr[i] = cdi10 * u_deg + cdi11 * v_deg + crpix1;
    }
}
}
