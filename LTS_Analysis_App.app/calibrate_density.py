from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
M_E = 9.1093837139e-31
C = 299_792_458.0
R_E = 2.8179403262e-15


def read_summary_value(path: str | Path, name: str) -> float:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["parameter"] == name:
                return float(row["value"])
    raise KeyError(f"{name!r} was not found in {path}")


def gas_density_from_pressure(pressure_pa: float, gas_temperature_k: float) -> float:
    return pressure_pa / (K_B * gas_temperature_k)


def differential_thomson_cross_section(scattering_angle_deg: float, polarization_factor: float = 1.0) -> float:
    theta = math.radians(scattering_angle_deg)
    unpolarized = 0.5 * R_E**2 * (1.0 + math.cos(theta) ** 2)
    return polarization_factor * unpolarized


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


def electron_temperature_from_width(
    sigma_pixel: float,
    nm_per_pixel: float,
    laser_wavelength_nm: float,
    scattering_angle_deg: float,
    instrument_sigma_pixel: float = 0.0,
) -> float:
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
    parser.add_argument("--raman-dsigma", type=float, required=True, help="Differential Raman cross section in m^2/sr")
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
    thomson_dsigma = differential_thomson_cross_section(args.scattering_angle_deg, args.polarization_factor)
    ne_m3 = electron_density_from_raman(
        thomson_area=thomson_area,
        raman_area=raman_area,
        gas_density_m3=gas_density,
        raman_differential_cross_section_m2_sr=args.raman_dsigma,
        thomson_differential_cross_section_m2_sr=thomson_dsigma,
        correction_factor=args.correction_factor,
    )

    print(f"area_kind: {args.area_kind}")
    print(f"thomson_area: {thomson_area:.6e} count*pixel")
    print(f"raman_area: {raman_area:.6e} count*pixel")
    print(f"hydrogen_density: {gas_density:.6e} m^-3")
    print(f"thomson_dsigma: {thomson_dsigma:.6e} m^2/sr")
    print(f"raman_dsigma: {args.raman_dsigma:.6e} m^2/sr")
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


if __name__ == "__main__":
    main()
