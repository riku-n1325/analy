from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


K_B = 1.380649e-23
EPSILON_0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_E = 9.1093837139e-31
C = 299_792_458.0
R_E = 2.8179403262e-15
PAPER_N2_STOKES_CROSS_SECTION_M2 = 3.82e-34
PAPER_THOMSON_CROSS_SECTION_M2 = R_E**2


def read_summary_value(path: str | Path, name: str) -> float:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["parameter"] == name:
                return float(row["value"])
    raise KeyError(f"{name!r} was not found in {path}")


def gas_density_from_pressure(pressure_pa: float, gas_temperature_k: float) -> float:
    if pressure_pa <= 0:
        raise ValueError("水素圧力は正の値を入力してください。")
    if gas_temperature_k <= 0:
        raise ValueError("気体温度は正の値を入力してください。0 Kは使えません。")
    return pressure_pa / (K_B * gas_temperature_k)


def differential_thomson_cross_section(scattering_angle_deg: float, polarization_factor: float = 1.0) -> float:
    theta = math.radians(scattering_angle_deg)
    unpolarized = 0.5 * R_E**2 * (1.0 + math.cos(theta) ** 2)
    return polarization_factor * unpolarized


def throughput_from_raman(
    raman_stokes_counts: float,
    gas_density_m3: float,
    raman_stokes_cross_section_m2: float,
    raman_shots: float = 1.0,
) -> float:
    """Paper Eq. (8): N_Stokes = k * n_gas * sigma_Stokes."""
    if raman_shots <= 0:
        raise ValueError("raman_shots must be positive.")
    if gas_density_m3 <= 0:
        raise ValueError("gas_density_m3 must be positive.")
    if raman_stokes_cross_section_m2 <= 0:
        raise ValueError("raman_stokes_cross_section_m2 must be positive.")
    counts_per_shot = raman_stokes_counts / raman_shots
    if counts_per_shot <= 0:
        raise ValueError("raman_stokes_counts per shot must be positive.")
    return counts_per_shot / (gas_density_m3 * raman_stokes_cross_section_m2)


def electron_density_from_thomson_counts(
    thomson_counts: float,
    throughput_k_m: float,
    thomson_cross_section_m2: float,
    thomson_shots: float = 1.0,
    correction_factor: float = 1.0,
) -> float:
    """Paper Eq. (3): N_T = k * n_e * sigma_T."""
    if thomson_shots <= 0:
        raise ValueError("thomson_shots must be positive.")
    if throughput_k_m <= 0:
        raise ValueError("throughput_k_m must be positive.")
    if thomson_cross_section_m2 <= 0:
        raise ValueError("thomson_cross_section_m2 must be positive.")
    if correction_factor <= 0:
        raise ValueError("correction_factor must be positive.")
    counts_per_shot = thomson_counts / thomson_shots
    if counts_per_shot <= 0:
        raise ValueError("thomson_counts per shot must be positive.")
    return correction_factor * counts_per_shot / (throughput_k_m * thomson_cross_section_m2)


def electron_density_from_paper_raman_calibration(
    thomson_counts: float,
    raman_stokes_counts: float,
    gas_density_m3: float,
    raman_stokes_cross_section_m2: float,
    thomson_cross_section_m2: float,
    thomson_shots: float = 1.0,
    raman_shots: float = 1.0,
    correction_factor: float = 1.0,
) -> tuple[float, float]:
    k_m = throughput_from_raman(
        raman_stokes_counts=raman_stokes_counts,
        gas_density_m3=gas_density_m3,
        raman_stokes_cross_section_m2=raman_stokes_cross_section_m2,
        raman_shots=raman_shots,
    )
    ne_m3 = electron_density_from_thomson_counts(
        thomson_counts=thomson_counts,
        throughput_k_m=k_m,
        thomson_cross_section_m2=thomson_cross_section_m2,
        thomson_shots=thomson_shots,
        correction_factor=correction_factor,
    )
    return ne_m3, k_m


def electron_density_from_raman(
    thomson_area: float,
    raman_area: float,
    gas_density_m3: float,
    raman_differential_cross_section_m2_sr: float,
    thomson_differential_cross_section_m2_sr: float,
    correction_factor: float = 1.0,
) -> float:
    if raman_area <= 0:
        raise ValueError("raman_area must be positive.")
    if thomson_differential_cross_section_m2_sr <= 0:
        raise ValueError("thomson_differential_cross_section_m2_sr must be positive.")

    return (
        gas_density_m3
        * (thomson_area / raman_area)
        * (raman_differential_cross_section_m2_sr / thomson_differential_cross_section_m2_sr)
        * correction_factor
    )


def scattering_wave_number(laser_wavelength_nm: float, scattering_angle_deg: float) -> float:
    if laser_wavelength_nm <= 0:
        raise ValueError("レーザー波長は正の値を入力してください。")
    laser_wavelength_m = laser_wavelength_nm * 1e-9
    theta = math.radians(scattering_angle_deg)
    sin_half = math.sin(theta / 2.0)
    if sin_half <= 0:
        raise ValueError("散乱角は0度より大きい値を入力してください。")
    return 4.0 * math.pi * sin_half / laser_wavelength_m


def debye_length_m(electron_temperature_ev: float, electron_density_m3: float) -> float:
    if electron_temperature_ev <= 0:
        raise ValueError("electron_temperature_ev must be positive.")
    if electron_density_m3 <= 0:
        raise ValueError("electron_density_m3 must be positive.")
    return math.sqrt(EPSILON_0 * electron_temperature_ev * E_CHARGE / (electron_density_m3 * E_CHARGE**2))


def scattering_parameter_alpha(
    electron_density_m3: float,
    electron_temperature_ev: float,
    laser_wavelength_nm: float,
    scattering_angle_deg: float,
) -> float:
    k_wave = scattering_wave_number(laser_wavelength_nm, scattering_angle_deg)
    lambda_d = debye_length_m(electron_temperature_ev, electron_density_m3)
    return 1.0 / (k_wave * lambda_d)


def electron_temperature_from_width(
    sigma_pixel: float,
    nm_per_pixel: float,
    laser_wavelength_nm: float,
    scattering_angle_deg: float,
    instrument_sigma_pixel: float = 0.0,
) -> float:
    if nm_per_pixel <= 0:
        raise ValueError("波長校正は正の値を入力してください。")
    if laser_wavelength_nm <= 0:
        raise ValueError("レーザー波長は正の値を入力してください。")
    sigma_pixel_physical = math.sqrt(max(sigma_pixel**2 - instrument_sigma_pixel**2, 0.0))
    sigma_lambda_m = sigma_pixel_physical * nm_per_pixel * 1e-9
    laser_wavelength_m = laser_wavelength_nm * 1e-9
    theta = math.radians(scattering_angle_deg)
    denominator = 2.0 * laser_wavelength_m * math.sin(theta / 2.0)
    if denominator <= 0:
        raise ValueError("scattering_angle_deg must be greater than 0.")

    ratio = sigma_lambda_m / denominator
    temperature_joule = M_E * C**2 * ratio**2
    return temperature_joule / E_CHARGE


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate electron density using Raman-calibrated Thomson scattering.")
    parser.add_argument("--thomson-summary", required=True, help="CSV made by fit_peak.py for Thomson spectrum")
    parser.add_argument("--raman-summary", required=True, help="CSV made by fit_peak.py for Raman spectrum")
    parser.add_argument("--area-kind", choices=["gaussian", "direct"], default="gaussian")
    parser.add_argument("--pressure-pa", type=float, required=True, help="Hydrogen pressure for Raman calibration")
    parser.add_argument("--gas-temperature-k", type=float, default=300.0)
    parser.add_argument("--raman-dsigma", type=float, default=PAPER_N2_STOKES_CROSS_SECTION_M2, help="Effective Stokes Raman cross section in m^2")
    parser.add_argument("--thomson-dsigma", type=float, default=PAPER_THOMSON_CROSS_SECTION_M2, help="Thomson differential cross section in m^2")
    parser.add_argument("--raman-shots", type=float, default=1.0)
    parser.add_argument("--thomson-shots", type=float, default=1.0)
    parser.add_argument("--scattering-angle-deg", type=float, required=True)
    parser.add_argument("--polarization-factor", type=float, default=1.0)
    parser.add_argument("--correction-factor", type=float, default=1.0, help="Transmission, laser energy, gate, and gain correction")
    parser.add_argument("--nm-per-pixel", type=float, default=None, help="Wavelength calibration for Te")
    parser.add_argument("--laser-wavelength-nm", type=float, default=None, help="Laser wavelength for Te")
    parser.add_argument("--instrument-sigma-pixel", type=float, default=0.0)
    args = parser.parse_args()

    area_name = "gaussian_area" if args.area_kind == "gaussian" else "direct_area"
    thomson_area = read_summary_value(args.thomson_summary, area_name)
    raman_area = read_summary_value(args.raman_summary, area_name)

    gas_density = gas_density_from_pressure(args.pressure_pa, args.gas_temperature_k)
    ne_m3, k_m = electron_density_from_paper_raman_calibration(
        thomson_counts=thomson_area,
        raman_stokes_counts=raman_area,
        gas_density_m3=gas_density,
        raman_stokes_cross_section_m2=args.raman_dsigma,
        thomson_cross_section_m2=args.thomson_dsigma,
        thomson_shots=args.thomson_shots,
        raman_shots=args.raman_shots,
        correction_factor=args.correction_factor,
    )

    print(f"area_kind: {args.area_kind}")
    print(f"thomson_area: {thomson_area:.6e} count*pixel")
    print(f"raman_area: {raman_area:.6e} count*pixel")
    print(f"hydrogen_density: {gas_density:.6e} m^-3")
    print(f"throughput_k: {k_m:.6e} m")
    print(f"thomson_dsigma: {args.thomson_dsigma:.6e} m^2")
    print(f"raman_stokes_sigma: {args.raman_dsigma:.6e} m^2")
    print(f"electron_density: {ne_m3:.6e} m^-3")
    print(f"electron_density: {ne_m3 / 1e6:.6e} cm^-3")

    if args.nm_per_pixel is not None and args.laser_wavelength_nm is not None:
        sigma_pixel = read_summary_value(args.thomson_summary, "sigma")
        te_ev = electron_temperature_from_width(
            sigma_pixel=sigma_pixel,
            nm_per_pixel=args.nm_per_pixel,
            laser_wavelength_nm=args.laser_wavelength_nm,
            scattering_angle_deg=args.scattering_angle_deg,
            instrument_sigma_pixel=args.instrument_sigma_pixel,
        )
        print(f"electron_temperature: {te_ev:.6e} eV")
        alpha = scattering_parameter_alpha(ne_m3, te_ev, args.laser_wavelength_nm, args.scattering_angle_deg)
        print(f"scattering_alpha: {alpha:.6e}")


if __name__ == "__main__":
    main()
