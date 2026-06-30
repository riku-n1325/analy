from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_spe import read_spe, spectrum_from_image  # noqa: E402


@dataclass
class PeakFit:
    center_pixel: float
    sigma_pixel: float
    fwhm_pixel: float
    amplitude: float
    baseline: float
    gaussian_area: float
    direct_area: float
    fit_min_pixel: int
    fit_max_pixel: int
    r_squared: float
    fit_points: int
    mask_min_pixel: int | None = None
    mask_max_pixel: int | None = None
    fixed_center_used: bool = False


def estimate_baseline(spectrum: np.ndarray, peak_pixel: int, window: int, sideband: int) -> float:
    left0 = max(0, peak_pixel - window - sideband)
    left1 = max(0, peak_pixel - window)
    right0 = min(len(spectrum), peak_pixel + window + 1)
    right1 = min(len(spectrum), peak_pixel + window + sideband + 1)

    samples = []
    if left1 > left0:
        samples.append(spectrum[left0:left1])
    if right1 > right0:
        samples.append(spectrum[right0:right1])
    if not samples:
        edge = min(100, len(spectrum) // 5)
        samples = [spectrum[:edge], spectrum[-edge:]]
    return float(np.median(np.concatenate(samples)))


def fit_gaussian_log_parabola(
    spectrum: np.ndarray,
    peak_pixel: int | None = None,
    window: int = 25,
    sideband: int = 80,
    threshold_fraction: float = 0.08,
    mask_min: int | None = None,
    mask_max: int | None = None,
    fixed_center: float | None = None,
) -> PeakFit:
    y_raw = spectrum.astype(np.float64)
    if peak_pixel is None:
        peak_pixel = int(np.argmax(y_raw))
    if fixed_center is not None:
        peak_pixel = int(round(fixed_center))

    baseline = estimate_baseline(y_raw, peak_pixel, window, sideband)
    x0 = max(0, peak_pixel - window)
    x1 = min(len(y_raw), peak_pixel + window + 1)

    x = np.arange(x0, x1, dtype=np.float64)
    y = y_raw[x0:x1] - baseline
    y = np.clip(y, 0, None)

    peak_height = float(np.max(y))
    if peak_height <= 0:
        raise ValueError("Peak height after baseline subtraction is not positive.")

    use = y > peak_height * threshold_fraction
    if mask_min is not None and mask_max is not None:
        use &= ~((x >= mask_min) & (x <= mask_max))
    if np.count_nonzero(use) < 5:
        use = y > 0
        if mask_min is not None and mask_max is not None:
            use &= ~((x >= mask_min) & (x <= mask_max))
    if np.count_nonzero(use) < 5:
        raise ValueError("Not enough positive points around the peak for Gaussian fitting.")

    # Fit log(y) = a*z^2 + b*z + c. This is a Gaussian after baseline subtraction.
    z = x[use] - peak_pixel
    log_y = np.log(y[use])
    if fixed_center is not None:
        design = np.column_stack([z**2, np.ones_like(z)])
        a, c = np.linalg.lstsq(design, log_y, rcond=None)[0]
        if a >= 0:
            weights = y[use]
            center = float(fixed_center)
            sigma = float(np.sqrt(np.sum(weights * (x[use] - center) ** 2) / np.sum(weights)))
            amplitude = peak_height
        else:
            center = float(fixed_center)
            sigma = float(math.sqrt(-1.0 / (2.0 * a)))
            amplitude = float(math.exp(c))
    else:
        a, b, c = np.polyfit(z, log_y, deg=2)
        if a >= 0:
            # Fallback to moment estimates if the logarithmic fit is not concave.
            weights = y[use]
            center = float(np.sum(x[use] * weights) / np.sum(weights))
            sigma = float(np.sqrt(np.sum(weights * (x[use] - center) ** 2) / np.sum(weights)))
            amplitude = peak_height
        else:
            sigma = float(math.sqrt(-1.0 / (2.0 * a)))
            center = float(peak_pixel - b / (2.0 * a))
            amplitude = float(math.exp(c - a * (center - peak_pixel) ** 2))

    fwhm = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma)
    gaussian_area = float(amplitude * sigma * math.sqrt(2.0 * math.pi))
    direct_area = float(np.sum(y))
    y_fit_used = amplitude * np.exp(-0.5 * ((x[use] - center) / sigma) ** 2)
    ss_res = float(np.sum((y[use] - y_fit_used) ** 2))
    ss_tot = float(np.sum((y[use] - np.mean(y[use])) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return PeakFit(
        center_pixel=center,
        sigma_pixel=sigma,
        fwhm_pixel=fwhm,
        amplitude=amplitude,
        baseline=baseline,
        gaussian_area=gaussian_area,
        direct_area=direct_area,
        fit_min_pixel=x0,
        fit_max_pixel=x1 - 1,
        r_squared=r_squared,
        fit_points=int(np.count_nonzero(use)),
        mask_min_pixel=mask_min,
        mask_max_pixel=mask_max,
        fixed_center_used=fixed_center is not None,
    )


def gaussian(x: np.ndarray, fit: PeakFit) -> np.ndarray:
    return fit.baseline + fit.amplitude * np.exp(-0.5 * ((x - fit.center_pixel) / fit.sigma_pixel) ** 2)


def save_fit_csv(path: Path, spectrum: np.ndarray, fit: PeakFit) -> None:
    x = np.arange(len(spectrum), dtype=np.float64)
    y_fit = gaussian(x, fit)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel", "counts", "gaussian_fit_counts"])
        for pixel, counts, fit_counts in zip(x.astype(int), spectrum, y_fit):
            writer.writerow([pixel, float(counts), float(fit_counts)])


def save_summary_csv(path: Path, fit: PeakFit) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "unit"])
        writer.writerow(["center", fit.center_pixel, "pixel"])
        writer.writerow(["sigma", fit.sigma_pixel, "pixel"])
        writer.writerow(["fwhm", fit.fwhm_pixel, "pixel"])
        writer.writerow(["amplitude", fit.amplitude, "counts"])
        writer.writerow(["baseline", fit.baseline, "counts"])
        writer.writerow(["gaussian_area", fit.gaussian_area, "count_pixel"])
        writer.writerow(["direct_area", fit.direct_area, "count_pixel"])
        writer.writerow(["fit_min", fit.fit_min_pixel, "pixel"])
        writer.writerow(["fit_max", fit.fit_max_pixel, "pixel"])
        writer.writerow(["r_squared", fit.r_squared, ""])
        writer.writerow(["fit_points", fit.fit_points, "pixel_count"])
        writer.writerow(["mask_min", "" if fit.mask_min_pixel is None else fit.mask_min_pixel, "pixel"])
        writer.writerow(["mask_max", "" if fit.mask_max_pixel is None else fit.mask_max_pixel, "pixel"])
        writer.writerow(["fixed_center_used", int(fit.fixed_center_used), "0_or_1"])


def save_fit_png(path: Path, spectrum: np.ndarray, fit: PeakFit) -> None:
    width, height = 1200, 700
    ml, mr, mt, mb = 80, 30, 30, 95
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    x = np.arange(len(spectrum), dtype=np.float64)
    y_raw = spectrum.astype(np.float64)
    y_fit = gaussian(x, fit)

    lo = float(np.percentile(y_raw, 1))
    hi = float(max(np.percentile(y_raw, 99.8), y_fit.max()))
    if hi <= lo:
        hi = lo + 1

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[mt : mt + plot_h, ml : ml + plot_w] = 248
    canvas[mt + plot_h, ml : ml + plot_w + 1] = 30
    canvas[mt : mt + plot_h + 1, ml] = 30

    def to_xy(y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = np.linspace(ml, ml + plot_w - 1, len(y_values)).astype(int)
        py = mt + plot_h - 1 - np.clip((y_values - lo) / (hi - lo), 0, 1) * (plot_h - 1)
        return px, py.astype(int)

    def draw_line(y_values: np.ndarray, color: tuple[int, int, int]) -> None:
        xs, ys = to_xy(y_values)
        for i in range(len(xs) - 1):
            steps = max(abs(xs[i + 1] - xs[i]), abs(ys[i + 1] - ys[i]), 1)
            line_x = np.linspace(xs[i], xs[i + 1], steps + 1).astype(int)
            line_y = np.linspace(ys[i], ys[i + 1], steps + 1).astype(int)
            canvas[line_y, line_x] = color

    draw_line(y_raw, (20, 90, 180))
    draw_line(y_fit, (210, 70, 45))

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lines = [
        "blue: data, red: Gaussian fit",
        f"center = {fit.center_pixel:.2f} pixel",
        f"FWHM = {fit.fwhm_pixel:.2f} pixel",
        f"Gaussian area = {fit.gaussian_area:.3e} count*pixel",
        f"Direct area = {fit.direct_area:.3e} count*pixel",
        f"R^2 = {fit.r_squared:.4f}, fit points = {fit.fit_points}",
    ]
    y_text = height - mb + 15
    for line in lines:
        draw.text((ml, y_text), line, fill=(20, 20, 20), font=font)
        y_text += 14
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a single spectral peak with a Gaussian approximation.")
    parser.add_argument("spe_file", help="Path to .spe file")
    parser.add_argument("--out-dir", default=".", help="Directory for fit outputs")
    parser.add_argument("--y-min", type=int, default=None, help="First detector row to integrate")
    parser.add_argument("--y-max", type=int, default=None, help="One past last detector row to integrate")
    parser.add_argument("--peak-pixel", type=int, default=None, help="Approximate peak pixel. Default: brightest pixel.")
    parser.add_argument("--window", type=int, default=25, help="Half-width of fit region in pixels")
    parser.add_argument("--sideband", type=int, default=80, help="Width of each baseline sideband in pixels")
    parser.add_argument("--mask-min", type=int, default=None, help="First blocked pixel to exclude from fit")
    parser.add_argument("--mask-max", type=int, default=None, help="Last blocked pixel to exclude from fit")
    parser.add_argument("--fixed-center", type=float, default=None, help="Fix Gaussian center for a clipped peak")
    args = parser.parse_args()

    spe = read_spe(args.spe_file)
    spectrum = spectrum_from_image(spe.image, args.y_min, args.y_max)
    fit = fit_gaussian_log_parabola(
        spectrum,
        peak_pixel=args.peak_pixel,
        window=args.window,
        sideband=args.sideband,
        mask_min=args.mask_min,
        mask_max=args.mask_max,
        fixed_center=args.fixed_center,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = spe.path.stem
    fit_csv = out_dir / f"{stem}_fit_curve.csv"
    summary_csv = out_dir / f"{stem}_fit_summary.csv"
    fit_png = out_dir / f"{stem}_fit.png"

    save_fit_csv(fit_csv, spectrum, fit)
    save_summary_csv(summary_csv, fit)
    save_fit_png(fit_png, spectrum, fit)

    print(f"file: {spe.path}")
    print(f"center_pixel: {fit.center_pixel:.4f}")
    print(f"fwhm_pixel: {fit.fwhm_pixel:.4f}")
    print(f"baseline_counts: {fit.baseline:.4f}")
    print(f"gaussian_area_count_pixel: {fit.gaussian_area:.6e}")
    print(f"direct_area_count_pixel: {fit.direct_area:.6e}")
    print(f"r_squared: {fit.r_squared:.6f}")
    print(f"fit_points: {fit.fit_points}")
    print(f"fit curve CSV: {fit_csv}")
    print(f"fit summary CSV: {summary_csv}")
    print(f"fit PNG: {fit_png}")


if __name__ == "__main__":
    main()
