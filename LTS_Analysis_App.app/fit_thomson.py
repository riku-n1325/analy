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
class ThomsonFit:
    center_pixel: float
    sigma_pixel: float
    fwhm_pixel: float
    amplitude: float
    gaussian_area: float
    direct_area: float
    fit_min_pixel: int
    fit_max_pixel: int
    baseline_left: float
    baseline_right: float
    median_width: int
    fixed_center_used: bool
    r_squared: float
    fit_points: int


def moving_median(y: np.ndarray, width: int) -> np.ndarray:
    if width < 1:
        raise ValueError("median width must be positive.")
    if width % 2 == 0:
        width += 1
    half = width // 2
    out = np.empty_like(y, dtype=np.float64)
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i - half) : min(len(y), i + half + 1)])
    return out


def linear_baseline(
    x: np.ndarray,
    y: np.ndarray,
    left_min: int,
    left_max: int,
    right_min: int,
    right_max: int,
) -> np.ndarray:
    left = y[left_min : left_max + 1]
    right = y[right_min : right_max + 1]
    if len(left) < 3 or len(right) < 3:
        raise ValueError("Baseline sidebands must contain at least 3 pixels each.")

    x_left = (left_min + left_max) / 2.0
    x_right = (right_min + right_max) / 2.0
    if x_right == x_left:
        raise ValueError(
            "トムソン背景の左範囲と右範囲の代表x座標が同じです。"
            "背景左最小/最大と背景右最小/最大が別の範囲になるように設定してください。"
        )
    y_left = float(np.median(left))
    y_right = float(np.median(right))

    slope = (y_right - y_left) / (x_right - x_left)
    intercept = y_left - slope * x_left
    return slope * x + intercept


def _least_squares_lm(
    model,
    initial: np.ndarray,
    y: np.ndarray,
    max_iter: int = 80,
) -> np.ndarray:
    params = initial.astype(np.float64).copy()
    damping = 1e-3
    best_residual = model(params) - y
    best_sse = float(np.sum(best_residual**2))
    for _ in range(max_iter):
        residual = model(params) - y
        jac = np.empty((y.size, params.size), dtype=np.float64)
        for col in range(params.size):
            step = 1e-5 * (abs(params[col]) + 1.0)
            shifted = params.copy()
            shifted[col] += step
            jac[:, col] = (model(shifted) - model(params)) / step
        lhs = jac.T @ jac + damping * np.eye(params.size)
        rhs = -jac.T @ residual
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        trial = params + delta
        trial_residual = model(trial) - y
        trial_sse = float(np.sum(trial_residual**2))
        if trial_sse < best_sse:
            params = trial
            best_sse = trial_sse
            best_residual = trial_residual
            damping *= 0.35
            if np.linalg.norm(delta) < 1e-7 * (np.linalg.norm(params) + 1.0):
                break
        else:
            damping *= 3.0
    return params


def fit_broad_gaussian(
    spectrum: np.ndarray,
    fit_min: int,
    fit_max: int,
    baseline_left_min: int,
    baseline_left_max: int,
    baseline_right_min: int,
    baseline_right_max: int,
    median_width: int = 21,
    threshold_fraction: float = 0.15,
    fixed_center: float | None = None,
    mask_min: int | None = None,
    mask_max: int | None = None,
) -> tuple[ThomsonFit, np.ndarray, np.ndarray, np.ndarray]:
    raw = spectrum.astype(np.float64)
    smooth = moving_median(raw, median_width)
    x_all = np.arange(len(raw), dtype=np.float64)
    baseline = linear_baseline(
        x_all,
        smooth,
        baseline_left_min,
        baseline_left_max,
        baseline_right_min,
        baseline_right_max,
    )

    x = x_all[fit_min : fit_max + 1]
    y = smooth[fit_min : fit_max + 1] - baseline[fit_min : fit_max + 1]
    y = np.clip(y, 0.0, None)

    if np.max(y) <= 0:
        raise ValueError("No positive Thomson peak after baseline subtraction.")

    peak_index = int(np.argmax(y))
    peak_x = float(x[peak_index] if fixed_center is None else fixed_center)
    peak_height = float(y[peak_index])
    use = y > peak_height * threshold_fraction
    if np.count_nonzero(use) < 8:
        use = y > 0
    if mask_min is not None and mask_max is not None:
        use &= ~((x >= mask_min) & (x <= mask_max))
    if np.count_nonzero(use) < 8:
        raise ValueError("Not enough points to fit the Thomson peak.")

    z = x[use] - peak_x
    log_y = np.log(y[use])
    if fixed_center is not None:
        # With the laser/Rayleigh center known, fit log(y) = a*(x-center)^2 + c.
        design = np.column_stack([z**2, np.ones_like(z)])
        a, c = np.linalg.lstsq(design, log_y, rcond=None)[0]
        if a >= 0:
            weights = y
            center = float(fixed_center)
            sigma = float(np.sqrt(np.sum(weights * (x - center) ** 2) / np.sum(weights)))
            amplitude = peak_height
        else:
            center = float(fixed_center)
            sigma = float(math.sqrt(-1.0 / (2.0 * a)))
            amplitude = float(math.exp(c))
    else:
        a, b, c = np.polyfit(z, log_y, deg=2)
        if a >= 0:
            weights = y
            center = float(np.sum(x * weights) / np.sum(weights))
            sigma = float(np.sqrt(np.sum(weights * (x - center) ** 2) / np.sum(weights)))
            amplitude = peak_height
        else:
            sigma = float(math.sqrt(-1.0 / (2.0 * a)))
            center = float(peak_x - b / (2.0 * a))
            amplitude = float(math.exp(c - a * (center - peak_x) ** 2))

    fit_x = x[use]
    fit_y = y[use]
    if fixed_center is not None:
        def model(params: np.ndarray) -> np.ndarray:
            amp, log_sigma = params
            sigma_value = math.exp(float(log_sigma)) + 1e-12
            return amp * np.exp(-0.5 * ((fit_x - float(fixed_center)) / sigma_value) ** 2)

        refined = _least_squares_lm(
            model,
            np.array([amplitude, math.log(max(sigma, 1e-6))], dtype=np.float64),
            fit_y,
        )
        amplitude = float(refined[0])
        center = float(fixed_center)
        sigma = float(math.exp(float(refined[1])) + 1e-12)
    else:
        def model(params: np.ndarray) -> np.ndarray:
            amp, center_value, log_sigma = params
            sigma_value = math.exp(float(log_sigma)) + 1e-12
            return amp * np.exp(-0.5 * ((fit_x - center_value) / sigma_value) ** 2)

        refined = _least_squares_lm(
            model,
            np.array([amplitude, center, math.log(max(sigma, 1e-6))], dtype=np.float64),
            fit_y,
        )
        amplitude = float(refined[0])
        center = float(refined[1])
        sigma = float(math.exp(float(refined[2])) + 1e-12)
    if amplitude <= 0 or sigma <= 0:
        raise ValueError("非線形最小二乗フィットで正の振幅・幅を得られませんでした。")

    fwhm = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma)
    gaussian_area = float(amplitude * sigma * math.sqrt(2.0 * math.pi))
    direct_area = float(np.sum(y))
    y_fit_used = amplitude * np.exp(-0.5 * ((x[use] - center) / sigma) ** 2)
    ss_res = float(np.sum((y[use] - y_fit_used) ** 2))
    ss_tot = float(np.sum((y[use] - np.mean(y[use])) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    fit = ThomsonFit(
        center_pixel=center,
        sigma_pixel=sigma,
        fwhm_pixel=fwhm,
        amplitude=amplitude,
        gaussian_area=gaussian_area,
        direct_area=direct_area,
        fit_min_pixel=fit_min,
        fit_max_pixel=fit_max,
        baseline_left=float(np.median(smooth[baseline_left_min : baseline_left_max + 1])),
        baseline_right=float(np.median(smooth[baseline_right_min : baseline_right_max + 1])),
        median_width=median_width,
        fixed_center_used=fixed_center is not None,
        r_squared=r_squared,
        fit_points=int(np.count_nonzero(use)),
    )
    fit_curve = baseline + amplitude * np.exp(-0.5 * ((x_all - center) / sigma) ** 2)
    return fit, smooth, baseline, fit_curve


def save_summary_csv(path: Path, fit: ThomsonFit) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "unit"])
        writer.writerow(["center", fit.center_pixel, "pixel"])
        writer.writerow(["sigma", fit.sigma_pixel, "pixel"])
        writer.writerow(["fwhm", fit.fwhm_pixel, "pixel"])
        writer.writerow(["amplitude", fit.amplitude, "counts"])
        writer.writerow(["gaussian_area", fit.gaussian_area, "count_pixel"])
        writer.writerow(["direct_area", fit.direct_area, "count_pixel"])
        writer.writerow(["fit_min", fit.fit_min_pixel, "pixel"])
        writer.writerow(["fit_max", fit.fit_max_pixel, "pixel"])
        writer.writerow(["baseline_left", fit.baseline_left, "counts"])
        writer.writerow(["baseline_right", fit.baseline_right, "counts"])
        writer.writerow(["median_width", fit.median_width, "pixel"])
        writer.writerow(["fixed_center_used", int(fit.fixed_center_used), "0_or_1"])
        writer.writerow(["r_squared", fit.r_squared, ""])
        writer.writerow(["fit_points", fit.fit_points, "pixel_count"])


def save_curve_csv(path: Path, raw: np.ndarray, smooth: np.ndarray, baseline: np.ndarray, fit_curve: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel", "raw_counts", "smoothed_counts", "baseline_counts", "gaussian_fit_counts"])
        for i, values in enumerate(zip(raw, smooth, baseline, fit_curve)):
            writer.writerow([i, *[float(v) for v in values]])


def save_png(path: Path, raw: np.ndarray, smooth: np.ndarray, baseline: np.ndarray, fit_curve: np.ndarray, fit: ThomsonFit) -> None:
    width, height = 1250, 740
    ml, mr, mt, mb = 80, 35, 30, 115
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    lo = float(np.percentile(raw, 1))
    hi = float(max(np.percentile(raw, 99.5), np.max(fit_curve)))
    if hi <= lo:
        hi = lo + 1

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[mt : mt + plot_h, ml : ml + plot_w] = 248
    canvas[mt + plot_h, ml : ml + plot_w + 1] = 30
    canvas[mt : mt + plot_h + 1, ml] = 30

    def draw_line(y_values: np.ndarray, color: tuple[int, int, int]) -> None:
        xs = np.linspace(ml, ml + plot_w - 1, len(y_values)).astype(int)
        ys = mt + plot_h - 1 - np.clip((y_values - lo) / (hi - lo), 0, 1) * (plot_h - 1)
        ys = ys.astype(int)
        for i in range(len(xs) - 1):
            steps = max(abs(xs[i + 1] - xs[i]), abs(ys[i + 1] - ys[i]), 1)
            line_x = np.linspace(xs[i], xs[i + 1], steps + 1).astype(int)
            line_y = np.linspace(ys[i], ys[i + 1], steps + 1).astype(int)
            canvas[line_y, line_x] = color

    draw_line(raw, (180, 200, 230))
    draw_line(smooth, (20, 90, 180))
    draw_line(baseline, (80, 80, 80))
    draw_line(fit_curve, (210, 70, 45))

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    lines = [
        "pale blue: raw, blue: median-smoothed, gray: baseline, red: Gaussian fit",
        f"center = {fit.center_pixel:.2f} pixel",
        f"FWHM = {fit.fwhm_pixel:.2f} pixel",
        f"Gaussian area = {fit.gaussian_area:.3e} count*pixel",
        f"Direct area = {fit.direct_area:.3e} count*pixel",
        f"R^2 = {fit.r_squared:.4f}, fit points = {fit.fit_points}",
        f"fit range = {fit.fit_min_pixel}:{fit.fit_max_pixel}, median width = {fit.median_width} pixel",
    ]
    y_text = height - mb + 14
    for line in lines:
        draw.text((ml, y_text), line, fill=(20, 20, 20), font=font)
        y_text += 15
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a broad Thomson scattering spectrum.")
    parser.add_argument("spe_file", help="Path to .spe file")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--y-min", type=int, default=600)
    parser.add_argument("--y-max", type=int, default=970)
    parser.add_argument("--fit-min", type=int, default=360)
    parser.add_argument("--fit-max", type=int, default=760)
    parser.add_argument("--baseline-left-min", type=int, default=180)
    parser.add_argument("--baseline-left-max", type=int, default=330)
    parser.add_argument("--baseline-right-min", type=int, default=800)
    parser.add_argument("--baseline-right-max", type=int, default=1000)
    parser.add_argument("--mask-min", type=int, default=None, help="Excluded inverse-slit range minimum pixel")
    parser.add_argument("--mask-max", type=int, default=None, help="Excluded inverse-slit range maximum pixel")
    parser.add_argument("--median-width", type=int, default=21)
    parser.add_argument("--threshold-fraction", type=float, default=0.15)
    parser.add_argument("--fixed-center", type=float, default=None, help="Fix Gaussian center, e.g. Rayleigh peak pixel")
    args = parser.parse_args()

    spe = read_spe(args.spe_file)
    raw = spectrum_from_image(spe.image, args.y_min, args.y_max)
    fit, smooth, baseline, fit_curve = fit_broad_gaussian(
        raw,
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        baseline_left_min=args.baseline_left_min,
        baseline_left_max=args.baseline_left_max,
        baseline_right_min=args.baseline_right_min,
        baseline_right_max=args.baseline_right_max,
        median_width=args.median_width,
        threshold_fraction=args.threshold_fraction,
        fixed_center=args.fixed_center,
        mask_min=args.mask_min,
        mask_max=args.mask_max,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = spe.path.stem
    summary_path = out_dir / f"{stem}_thomson_fit_summary.csv"
    curve_path = out_dir / f"{stem}_thomson_fit_curve.csv"
    png_path = out_dir / f"{stem}_thomson_fit.png"

    save_summary_csv(summary_path, fit)
    save_curve_csv(curve_path, raw, smooth, baseline, fit_curve)
    save_png(png_path, raw, smooth, baseline, fit_curve, fit)

    print(f"file: {spe.path}")
    print(f"integrated rows: {args.y_min}:{args.y_max}")
    print(f"center_pixel: {fit.center_pixel:.4f}")
    print(f"sigma_pixel: {fit.sigma_pixel:.4f}")
    print(f"fwhm_pixel: {fit.fwhm_pixel:.4f}")
    print(f"gaussian_area_count_pixel: {fit.gaussian_area:.6e}")
    print(f"direct_area_count_pixel: {fit.direct_area:.6e}")
    print(f"r_squared: {fit.r_squared:.6f}")
    print(f"fit_points: {fit.fit_points}")
    print(f"summary CSV: {summary_path}")
    print(f"curve CSV: {curve_path}")
    print(f"fit PNG: {png_path}")


if __name__ == "__main__":
    main()
